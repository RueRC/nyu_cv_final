using UnityEngine;
using UnityEngine.AI;

[RequireComponent(typeof(NavMeshAgent))]
public class RunningPed : MonoBehaviour
{
    [Header("Corridor Selection")]
    public bool autoDetectCorridor = true;
    public Transform endAOverride;
    public Transform endBOverride;
    public float maxProbeDistance = 80f;
    [Range(8, 180)] public int halfCircleSamples = 72;
    public float edgeBackoff = 0.3f;

    [Header("Movement")]
    public float waypointTolerance = 0.35f;
    public bool immediateTurnAtEnd = true;
    public float overrideRunSpeed = 0f;

    [Header("Rotation (manual)")]
    public float turnResponsiveness = 6f;
    public float minVelocityForTurning = 0.05f;

    [Header("Animator")]
    public Animator animator;
    public int animatorLayer = 0;

    public string isRunningParam = "IsRunning";
    public string lookBackTrigger = "LookBack";

    public string runningStateName = "Running";
    public string lookBackStateName = "Run Look Back";

    [Header("LookBack (rare)")]
    [Range(0f, 1f)] public float lookBackChanceAtTurn = 0.08f;
    public float lookBackCooldown = 8.0f;

    [Tooltip("Only trigger look back if we have been moving for at least this long since last turn.")]
    public float minTravelTimeBeforeLookBack = 1.2f;

    private NavMeshAgent agent;
    private Vector3 endA, endB;
    private int currentTargetIndex = 0;
    private bool hasCorridor = false;

    private int runningHash;
    private float nextLookBackAllowedTime = 0f;
    private float lastTurnTime = -999f;
    private bool lookBackTriggeredThisLeg = false;

    void Awake()
    {
        agent = GetComponent<NavMeshAgent>();
        agent.updatePosition = true;
        agent.updateRotation = false;

        if (!animator) animator = GetComponentInChildren<Animator>();

        runningHash = Animator.StringToHash(runningStateName);
    }

    void Start()
    {
        if (overrideRunSpeed > 0f) agent.speed = overrideRunSpeed;

        if (autoDetectCorridor) hasCorridor = DetectCorridor(transform.position, out endA, out endB);
        else hasCorridor = TryUseManualEndpoints(out endA, out endB);

        if (!hasCorridor)
        {
            Debug.LogWarning($"[RunningPed] Cannot determine corridor endpoints. Disabling. ({name})");
            enabled = false;
            return;
        }

        currentTargetIndex =
            (Vector3.SqrMagnitude(endA - transform.position) <= Vector3.SqrMagnitude(endB - transform.position))
            ? 0 : 1;

        if (animator) animator.SetBool(isRunningParam, true);

        agent.isStopped = false;
        agent.ResetPath();
        ForceRepathTo(CurrentTarget());

        lastTurnTime = Time.time;
        lookBackTriggeredThisLeg = false;
    }

    void Update()
    {
        if (!agent.enabled) return;

        if (animator) animator.SetBool(isRunningParam, true);

        ManualRotateAlongVelocity();

        if (!agent.pathPending && HasReachedTarget())
        {
            TryTriggerLookBackOnTurn();

            ToggleTarget();
            agent.ResetPath();
            ForceRepathTo(CurrentTarget());

            lastTurnTime = Time.time;
            lookBackTriggeredThisLeg = false;
        }
    }

    private bool HasReachedTarget()
    {
        float reach = Mathf.Max(agent.stoppingDistance, waypointTolerance);
        return agent.remainingDistance <= reach;
    }

    private void ManualRotateAlongVelocity()
    {
        Vector3 dir = agent.desiredVelocity;
        dir.y = 0f;

        // Deadzone: ignore tiny direction changes
        if (dir.sqrMagnitude < 0.0004f) return; // ~0.02^2

        dir.Normalize();

        Quaternion targetRot = Quaternion.LookRotation(dir);

        // Limit turning speed (degrees per second)
        float maxDegPerSec = 180f; // try 120~240
        transform.rotation = Quaternion.RotateTowards(
            transform.rotation,
            targetRot,
            maxDegPerSec * Time.deltaTime
        );
    }


    private void TryTriggerLookBackOnTurn()
    {
        if (!animator) return;
        if (Time.time < nextLookBackAllowedTime) return;
        if (lookBackTriggeredThisLeg) return;

        if (Time.time - lastTurnTime < minTravelTimeBeforeLookBack)
            return;

        var cur = animator.GetCurrentAnimatorStateInfo(animatorLayer);
        if (cur.shortNameHash != runningHash)
            return;

        if (Random.value <= lookBackChanceAtTurn)
        {
            animator.ResetTrigger(lookBackTrigger);
            animator.SetTrigger(lookBackTrigger);

            lookBackTriggeredThisLeg = true;
            nextLookBackAllowedTime = Time.time + lookBackCooldown;
        }
    }

    private void ToggleTarget() => currentTargetIndex = 1 - currentTargetIndex;
    private Vector3 CurrentTarget() => (currentTargetIndex == 0) ? endA : endB;

    private void ForceRepathTo(Vector3 target)
    {
        if (!agent.enabled) return;

        if (NavMesh.SamplePosition(target, out var hit, 1.0f, NavMesh.AllAreas))
        {
            agent.SetDestination(hit.position);
        }
        else
        {
            Vector3 dir = (target - transform.position);
            if (dir.sqrMagnitude > 0.0001f)
            {
                Vector3 tryPos = target - dir.normalized * edgeBackoff;
                if (NavMesh.SamplePosition(tryPos, out var hit2, 1.5f, NavMesh.AllAreas))
                    agent.SetDestination(hit2.position);
            }
        }
    }

    private bool TryUseManualEndpoints(out Vector3 A, out Vector3 B)
    {
        A = B = transform.position;
        if (!endAOverride || !endBOverride) return false;

        A = FlattenY(endAOverride.position);
        B = FlattenY(endBOverride.position);

        bool okA = NavMesh.SamplePosition(A, out var hA, 2f, NavMesh.AllAreas);
        bool okB = NavMesh.SamplePosition(B, out var hB, 2f, NavMesh.AllAreas);

        if (okA) A = hA.position;
        if (okB) B = hB.position;

        return okA && okB && (Vector3.Distance(A, B) > 1f);
    }

    private bool DetectCorridor(Vector3 origin, out Vector3 A, out Vector3 B)
    {
        origin = FlattenY(origin);
        A = B = origin;

        if (!NavMesh.SamplePosition(origin, out var startHit, 2f, NavMesh.AllAreas))
            return false;

        origin = startHit.position;
        float bestScore = -1f;

        for (int i = 0; i < halfCircleSamples; i++)
        {
            float angleDeg = (i + 0.5f) * (180f / halfCircleSamples);
            Vector3 dir = Quaternion.Euler(0f, angleDeg, 0f) * Vector3.forward;

            ProbeOneDirection(origin, dir, out float dFwd, out Vector3 pFwd);
            ProbeOneDirection(origin, -dir, out float dBack, out Vector3 pBack);

            float score = dFwd + dBack;
            if (score > bestScore)
            {
                bestScore = score;
                A = pBack;
                B = pFwd;
            }
        }

        Vector3 axis = (B - A);
        if (axis.sqrMagnitude < 1f) return false;

        Vector3 nAxis = axis.normalized;
        A = A + nAxis * edgeBackoff;
        B = B - nAxis * edgeBackoff;

        A = FlattenY(A);
        B = FlattenY(B);

        if (NavMesh.SamplePosition(A, out var ha, 1.5f, NavMesh.AllAreas)) A = ha.position;
        if (NavMesh.SamplePosition(B, out var hb, 1.5f, NavMesh.AllAreas)) B = hb.position;

        return Vector3.Distance(A, B) > 1.0f;
    }

    private void ProbeOneDirection(Vector3 origin, Vector3 dir, out float distance, out Vector3 endPoint)
    {
        dir.y = 0f;
        if (dir.sqrMagnitude < 1e-6f) dir = Vector3.forward;
        dir.Normalize();

        Vector3 far = origin + dir * maxProbeDistance;

        if (!NavMesh.SamplePosition(far, out var farHit, 2f, NavMesh.AllAreas))
        {
            float d = maxProbeDistance;
            bool found = false;

            for (int i = 0; i < 8; i++)
            {
                d *= 0.6f;
                Vector3 test = origin + dir * d;
                if (NavMesh.SamplePosition(test, out farHit, 2f, NavMesh.AllAreas))
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                distance = 0f;
                endPoint = origin;
                return;
            }
        }

        if (NavMesh.Raycast(origin, farHit.position, out var hit, NavMesh.AllAreas))
        {
            endPoint = hit.position;
            distance = hit.distance;
        }
        else
        {
            endPoint = farHit.position;
            distance = Vector3.Distance(origin, endPoint);
        }
    }

    private static Vector3 FlattenY(Vector3 v) => new Vector3(v.x, 0f, v.z);
}
