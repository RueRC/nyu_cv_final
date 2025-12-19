using UnityEngine;
using UnityEngine.AI;

[RequireComponent(typeof(NavMeshAgent))]
public class WanderPed : MonoBehaviour
{
    // =========================
    // Automatic wandering
    // =========================

    [Header("Automatic wandering")]
    public bool autoWander = true;                    // Whether the agent keeps wandering automatically
    public float wanderRadius = 20f;                  // Fallback random radius around current position
    public Vector2 intervalRange = new Vector2(1.5f, 3f); // Time interval range (seconds) before picking a new destination

    // =========================
    // Optional: bounding box constraint
    // =========================

    [Header("Optional: restrict movement within bounds")]
    public bool useBounds = false;                    // Enable bounding box constraint
    public Transform boundsCenter;                    // Center of the bounding box (can be an empty GameObject)
    public Vector3 boundsSize = new Vector3(60, 10, 60);  // XZ defines horizontal range

    // =========================
    // Long straight-line bias
    // =========================

    [Header("Long straight-line bias")]
    public bool preferStraight = true;

    [Tooltip("Preferred length (meters) of each straight segment.")]
    public float desiredStraightDistance = 25f;

    [Tooltip("Small heading noise (degrees). Smaller = straighter.")]
    public float headingJitterDegrees = 10f;

    [Tooltip("How many attempts to find a far reachable point in the desired direction.")]
    public int straightAttempts = 8;

    [Tooltip("Sample radius when projecting candidate points to the NavMesh.")]
    public float sampleRadius = 3f;

    // =========================
    // Hesitation / pause behavior
    // =========================

    [Header("Hesitation / pause behavior")]
    [Range(0f, 1f)]
    public float pauseProbabilityOnStuckOrObstacle = 0.35f;

    [Tooltip("Pause duration range (seconds). Uses realtime to avoid timeScale issues.")]
    public Vector2 pauseDurationRange = new Vector2(0.4f, 1.6f);

    [Tooltip("Forward obstacle check distance (meters).")]
    public float obstacleCheckDistance = 1.2f;

    [Tooltip("Sphere radius for obstacle detection.")]
    public float obstacleCheckRadius = 0.25f;

    [Tooltip("LayerMask used for obstacle detection (recommend: walls/buildings/static only).")]
    public LayerMask obstacleMask = ~0;

    // =========================
    // Debug
    // =========================

    [Header("Debug")]
    public bool drawDebug = false;

    private NavMeshAgent agent;
    private float nextPickTime;
    private float stuckTimer;

    private bool isPausing = false;
    private float pauseEndRealtime = 0f;

    void Awake()
    {
        agent = GetComponent<NavMeshAgent>();
        if (!agent)
            Debug.LogError("WanderAI requires a NavMeshAgent component on the same GameObject.");
    }

    void OnEnable()
    {
        ScheduleNextPick();
    }

    void Start()
    {
        // Ensure the agent starts on the NavMesh (helps when spawning at arbitrary positions)
        SnapToNavMesh(5f);

        // Pick an initial destination
        PickNewDestination();
        ScheduleNextPick();
    }

    void Update()
    {
        if (!autoWander || agent == null) return;

        // If we are not on the NavMesh, try to snap back and pick a new destination.
        if (!agent.isOnNavMesh)
        {
            SnapToNavMesh(5f);
            if (agent.isOnNavMesh)
            {
                ForceResumeAndRepath();
                ScheduleNextPick();
            }
            return;
        }

        // Handle pause state: when pause ends, we IMMEDIATELY replan and set a new destination.
        if (isPausing)
        {
            if (Time.realtimeSinceStartup >= pauseEndRealtime)
            {
                isPausing = false;
                ForceResumeAndRepath();   // <-- key requirement: re-set destination immediately after hesitation
                ScheduleNextPick();
            }
            return;
        }

        // Detect if we reached destination or timed out
        bool reached = !agent.pathPending && agent.remainingDistance <= Mathf.Max(agent.stoppingDistance, 0.2f);
        bool timeout = Time.time >= nextPickTime;

        // Simple stuck detection: velocity near zero for >2 seconds while having a path
        bool shouldBeMoving = agent.hasPath && !agent.pathPending && agent.remainingDistance > 1.0f;
        if (shouldBeMoving && agent.velocity.sqrMagnitude < 0.01f) stuckTimer += Time.deltaTime;
        else stuckTimer = 0f;

        // Obstacle ahead detection (optional but useful to trigger hesitation)
        bool obstacleAhead = CheckObstacleAhead(out RaycastHit hit);

        if (drawDebug && obstacleAhead)
            Debug.DrawLine(transform.position + Vector3.up * 0.6f, hit.point, Color.red);

        // If reached/timeout/stuck/obstacle -> either pause (probabilistically) or replan immediately
        if (reached || timeout || stuckTimer > 2f || obstacleAhead)
        {
            bool triggerPause = Random.value < pauseProbabilityOnStuckOrObstacle;

            if (triggerPause)
            {
                BeginPause(Random.Range(pauseDurationRange.x, pauseDurationRange.y));
                // IMPORTANT: We do NOT leave the agent permanently stopped.
                // On pause end, ForceResumeAndRepath() will be called.
            }
            else
            {
                ForceResumeAndRepath();
                ScheduleNextPick();
            }

            stuckTimer = 0f;
        }

        if (drawDebug && agent.hasPath)
        {
            var path = agent.path;
            for (int i = 0; i < path.corners.Length - 1; i++)
                Debug.DrawLine(path.corners[i] + Vector3.up * 0.1f, path.corners[i + 1] + Vector3.up * 0.1f, Color.green);
        }
    }

    /// <summary>
    /// Starts a hesitation pause using realtime clock (independent of Time.timeScale).
    /// </summary>
    private void BeginPause(float seconds)
    {
        isPausing = true;
        pauseEndRealtime = Time.realtimeSinceStartup + seconds;

        // Stop agent movement during the hesitation
        agent.isStopped = true;
        agent.ResetPath(); // drop current path to avoid getting "stuck in a bad path state"
    }

    /// <summary>
    /// Forcefully resumes the agent and immediately sets a new destination.
    /// </summary>
    private void ForceResumeAndRepath()
    {
        agent.isStopped = false;
        agent.ResetPath();           // ensure clean state
        SnapToNavMesh(2f);           // helps if small drift happened
        PickNewDestination();        // ALWAYS sets a destination
    }

    /// <summary>
    /// Picks a new destination on the NavMesh.
    /// Prefers long straight motion if enabled; otherwise falls back to random sampling.
    /// </summary>
    void PickNewDestination()
    {
        // 1) Try straight-line biased destination
        if (preferStraight)
        {
            if (TryPickStraightDestination(out Vector3 straightDest))
            {
                agent.SetDestination(straightDest);
                if (drawDebug)
                    Debug.DrawLine(transform.position + Vector3.up * 0.2f, straightDest + Vector3.up * 0.2f, Color.cyan, 1.0f);
                return;
            }
        }

        // 2) Fallback: original random destination logic
        Vector3 candidate;

        if (useBounds && boundsCenter)
        {
            Vector3 half = boundsSize * 0.5f;
            candidate = boundsCenter.position + new Vector3(
                Random.Range(-half.x, half.x),
                0f,
                Random.Range(-half.z, half.z)
            );
        }
        else
        {
            candidate = transform.position + Random.insideUnitSphere * wanderRadius;
            candidate.y = transform.position.y;
        }

        // Project candidate onto NavMesh
        if (NavMesh.SamplePosition(candidate, out var hit, 2f, agent.areaMask))
        {
            agent.SetDestination(hit.position);
            return;
        }

        // Final fallback: pick a random vertex from NavMesh triangulation
        var tri = NavMesh.CalculateTriangulation();
        if (tri.vertices != null && tri.vertices.Length > 0)
        {
            var v = tri.vertices[Random.Range(0, tri.vertices.Length)];
            agent.SetDestination(v);
        }
    }

    /// <summary>
    /// Attempts to pick a far reachable destination in (roughly) the current movement heading.
    /// This encourages long straight segments.
    /// </summary>
    private bool TryPickStraightDestination(out Vector3 dest)
    {
        dest = transform.position;

        // Determine the base heading: prefer current velocity direction, otherwise forward
        Vector3 heading = agent.velocity.sqrMagnitude > 0.02f ? agent.velocity.normalized : transform.forward;
        heading.y = 0f;
        if (heading.sqrMagnitude < 1e-6f) heading = Vector3.forward;

        // Add small jitter
        float jitter = Random.Range(-headingJitterDegrees, headingJitterDegrees);
        Vector3 dir = Quaternion.Euler(0f, jitter, 0f) * heading.normalized;

        // Try multiple distances (far to near) and a few angular variations
        for (int i = 0; i < straightAttempts; i++)
        {
            float t = 1.0f - (i / Mathf.Max(1f, straightAttempts - 1f)) * 0.6f; // 1.0 -> 0.4
            float d = desiredStraightDistance * t;

            Vector3 guess = transform.position + dir * d;

            // If bounds are enabled, clamp the guess into bounds on XZ
            if (useBounds && boundsCenter)
            {
                Vector3 half = boundsSize * 0.5f;
                Vector3 c = boundsCenter.position;
                guess.x = Mathf.Clamp(guess.x, c.x - half.x, c.x + half.x);
                guess.z = Mathf.Clamp(guess.z, c.z - half.z, c.z + half.z);
            }

            // Project onto NavMesh
            if (!NavMesh.SamplePosition(guess, out var hit, sampleRadius, agent.areaMask))
                continue;

            // Validate reachability
            var path = new NavMeshPath();
            if (agent.CalculatePath(hit.position, path) && path.status == NavMeshPathStatus.PathComplete)
            {
                dest = hit.position;
                return true;
            }

            // Slightly vary direction if not reachable
            float extra = Random.Range(-headingJitterDegrees, headingJitterDegrees);
            dir = Quaternion.Euler(0f, extra, 0f) * heading.normalized;
        }

        return false;
    }

    /// <summary>
    /// Schedules the next destination pick time.
    /// </summary>
    void ScheduleNextPick()
    {
        nextPickTime = Time.time + Random.Range(intervalRange.x, intervalRange.y);
    }

    /// <summary>
    /// Snaps the agent to the nearest NavMesh position.
    /// Uses disable/enable to avoid transform vs agent conflicts.
    /// </summary>
    private void SnapToNavMesh(float maxDist)
    {
        if (NavMesh.SamplePosition(transform.position, out var hit, maxDist, NavMesh.AllAreas))
        {
            bool wasEnabled = agent.enabled;
            agent.enabled = false;
            transform.position = hit.position;
            agent.enabled = wasEnabled;
        }
    }

    /// <summary>
    /// Checks if there is an obstacle directly ahead.
    /// Ignores hits on the agent itself.
    /// </summary>
    private bool CheckObstacleAhead(out RaycastHit hit)
    {
        Vector3 dir = agent.velocity.sqrMagnitude > 0.02f ? agent.velocity.normalized : transform.forward;
        dir.y = 0f;
        if (dir.sqrMagnitude < 1e-6f) dir = Vector3.forward;

        Vector3 origin = transform.position + Vector3.up * 0.6f;

        if (Physics.SphereCast(origin, obstacleCheckRadius, dir, out hit,
                               obstacleCheckDistance, obstacleMask, QueryTriggerInteraction.Ignore))
        {
            // Ignore self-hits
            if (hit.collider != null && hit.collider.transform.IsChildOf(transform))
                return false;

            return true;
        }

        return false;
    }

    /// <summary>
    /// Draws debug gizmos in the Scene view when the object is selected.
    /// </summary>
    void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.yellow;
        if (useBounds)
        {
            var center = boundsCenter ? boundsCenter.position : transform.position;
            Gizmos.DrawWireCube(center, boundsSize);
        }
        else
        {
            Gizmos.DrawWireSphere(transform.position, wanderRadius);
        }
    }
}
