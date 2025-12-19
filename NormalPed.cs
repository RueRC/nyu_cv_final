using System.Collections;
using UnityEngine;
using UnityEngine.AI;

[RequireComponent(typeof(NavMeshAgent))]
public class NormalPed : MonoBehaviour
{
    [Header("=== Corridor Selection ===")]
    public bool autoDetectCorridor = true;
    public Transform endAOverride;
    public Transform endBOverride;
    public float maxProbeDistance = 80f;
    [Range(8, 180)] public int halfCircleSamples = 72;
    public float edgeBackoff = 0.3f;

    [Header("=== Walking ===")]
    public float waypointTolerance = 0.3f;
    public bool immediateTurnAtEnd = true;

    [Header("=== Stop & Go ===")]
    public bool enableStopAndGo = true;
    public Vector2 moveDurationRange = new Vector2(3f, 10f);
    public Vector2 idleDurationRange = new Vector2(1f, 4f);

    [Header("=== Animator (Strict Text1 -> Text2, Fast Return to Walk) ===")]
    public Animator animator;
    public int animatorLayer = 0;

    [Tooltip("Trigger: Walking -> Texting While Standing")]
    public string startTextTrigger = "StartText";

    [Tooltip("Bool: Texting While Standing 0 -> Walking requires true (if you keep that transition).")]
    public string isWalkingParam = "IsWalking";

    // Your EXACT state names:
    public string walkingStateName = "Walking";
    public string texting1StateName = "Texting While Standing";
    public string texting2StateName = "Texting While Standing 0";

    [Tooltip("Text2 finish threshold (0.95 = cut last 5% for smoother transition).")]
    [Range(0.80f, 1.00f)]
    public float text2FinishThreshold = 0.95f;

    [Tooltip("Max seconds to wait for entering Text1 after trigger (prevents deadlock).")]
    public float enterText1Timeout = 2.0f;

    [Tooltip("Max seconds to wait for entering Text2 after forcing it (prevents deadlock).")]
    public float enterText2Timeout = 2.0f;

    [Tooltip("Max seconds to confirm we are in Walking before allowing movement.")]
    public float enterWalkingTimeout = 3.0f;

    [Tooltip("Tiny buffer seconds to avoid edge cases.")]
    public float timingBuffer = 0.02f;

    [Header("=== Gizmos ===")]
    public bool drawGizmos = true;
    public Color gizmoColor = new Color(0.2f, 0.9f, 0.9f, 0.9f);
    public float gizmoSphereRadius = 0.25f;

    private NavMeshAgent agent;
    private Vector3 endA, endB;
    private int currentTargetIndex = 0;
    private bool hasCorridor = false;

    private int walkingHash;
    private int text1Hash;
    private int text2Hash;

    void Awake()
    {
        agent = GetComponent<NavMeshAgent>();
        agent.updateRotation = true;
        agent.updatePosition = true;

        if (!animator) animator = GetComponentInChildren<Animator>();

        walkingHash = Animator.StringToHash(walkingStateName);
        text1Hash = Animator.StringToHash(texting1StateName);
        text2Hash = Animator.StringToHash(texting2StateName);
    }

    void Start()
    {
        if (autoDetectCorridor) hasCorridor = DetectCorridor(transform.position, out endA, out endB);
        else hasCorridor = TryUseManualEndpoints(out endA, out endB);

        if (!hasCorridor)
        {
            Debug.LogWarning($"[NormalPed] Cannot determine corridor endpoints. Disabling. ({name})");
            enabled = false;
            return;
        }

        currentTargetIndex = (Vector3.SqrMagnitude(endA - transform.position) <= Vector3.SqrMagnitude(endB - transform.position)) ? 0 : 1;

        ForceToWalkingNow();
        agent.isStopped = false;
        agent.ResetPath();
        ForceRepathTo(CurrentTarget());

        if (enableStopAndGo) StartCoroutine(StopAndGoLoop());
    }

    void Update()
    {
        if (!agent.enabled || agent.pathPending) return;
        if (enableStopAndGo) return;

        if (agent.remainingDistance <= Mathf.Max(agent.stoppingDistance, waypointTolerance))
        {
            ToggleTarget();
            if (immediateTurnAtEnd) ForceRepathTo(CurrentTarget());
            else StartCoroutine(ShortPauseThenMove());
        }
    }

    private IEnumerator ShortPauseThenMove()
    {
        agent.isStopped = true;
        agent.ResetPath();

        yield return PlayTextingSequence_StrictText1ThenText2_FastReturn();

        ForceToWalkingNow();
        yield return WaitUntilInWalkingOrTimeout();

        agent.isStopped = false;
        agent.ResetPath();
        ForceRepathTo(CurrentTarget());
    }

    private IEnumerator StopAndGoLoop()
    {
        while (true)
        {
            // ----- Move Phase -----
            float moveT = Random.Range(moveDurationRange.x, moveDurationRange.y);
            float timer = 0f;

            ForceToWalkingNow();
            yield return WaitUntilInWalkingOrTimeout();

            agent.isStopped = false;
            if (!agent.hasPath)
            {
                agent.ResetPath();
                ForceRepathTo(CurrentTarget());
            }

            while (timer < moveT)
            {
                if (!agent.pathPending &&
                    agent.remainingDistance <= Mathf.Max(agent.stoppingDistance, waypointTolerance))
                {
                    ToggleTarget();
                    ForceRepathTo(CurrentTarget());
                }

                timer += Time.deltaTime;
                yield return null;
            }

            // ----- Idle Phase -----
            float idleT = Random.Range(idleDurationRange.x, idleDurationRange.y);

            agent.isStopped = true;
            agent.ResetPath();

            // Start texting sequence immediately, but still keep a minimum idle time.
            Coroutine textingCo = StartCoroutine(PlayTextingSequence_StrictText1ThenText2_FastReturn());

            float minStopEndTime = Time.time + idleT;
            while (Time.time < minStopEndTime) yield return null;

            // Ensure texting has finished (text1 full, text2 to threshold)
            yield return textingCo;

            ForceToWalkingNow();
            yield return WaitUntilInWalkingOrTimeout();

            agent.isStopped = false;
            agent.ResetPath();
            ForceRepathTo(CurrentTarget());
        }
    }

    // ---------------- Animator control ----------------

    private void SetWalkingAnim(bool walking)
    {
        if (!animator) return;
        animator.SetBool(isWalkingParam, walking);
    }

    private void TriggerTexting()
    {
        if (!animator) return;
        animator.ResetTrigger(startTextTrigger);
        animator.SetTrigger(startTextTrigger);
    }

    private void ForceToWalkingNow()
    {
        if (!animator) return;

        animator.ResetTrigger(startTextTrigger);
        animator.SetBool(isWalkingParam, true);
        animator.CrossFadeInFixedTime(walkingStateName, 0f, animatorLayer);
    }

    /// <summary>
    /// Strict guarantee:
    /// - Text1: must fully finish (normalizedTime >= 1)
    /// - Text2: must reach text2FinishThreshold, then we immediately return to Walking (smooth + fast)
    /// </summary>
    private IEnumerator PlayTextingSequence_StrictText1ThenText2_FastReturn()
    {
        if (!animator) yield break;

        // Hold walking false during texting
        animator.SetBool(isWalkingParam, false);

        // Trigger Text1
        TriggerTexting();

        // Wait enter Text1 (best effort)
        yield return WaitUntilEnterShortHash(text1Hash, enterText1Timeout);

        // Wait Text1 finishes fully
        yield return WaitUntilShortHashFinishes(text1Hash, 1.0f);

        if (timingBuffer > 0f) yield return new WaitForSeconds(timingBuffer);

        // Force Text2 immediately (do not depend on Animator transition)
        animator.CrossFadeInFixedTime(texting2StateName, 0f, animatorLayer);

        // Wait enter Text2 (best effort), then wait until it reaches threshold
        yield return WaitUntilEnterShortHash(text2Hash, enterText2Timeout);
        yield return WaitUntilShortHashFinishes(text2Hash, text2FinishThreshold);

        if (timingBuffer > 0f) yield return new WaitForSeconds(timingBuffer);

        // Do NOT set walking true here; caller will ForceToWalkingNow() right after this coroutine ends.
    }

    private IEnumerator WaitUntilEnterShortHash(int shortHash, float timeoutSeconds)
    {
        if (!animator) yield break;

        float tEnd = Time.time + Mathf.Max(0.05f, timeoutSeconds);
        while (Time.time < tEnd)
        {
            var cur = animator.GetCurrentAnimatorStateInfo(animatorLayer);
            var nxt = animator.GetNextAnimatorStateInfo(animatorLayer);

            if (cur.shortNameHash == shortHash) yield break;
            if (animator.IsInTransition(animatorLayer) && nxt.shortNameHash == shortHash) yield break;

            yield return null;
        }
    }

    /// <summary>
    /// Wait until the given state's normalizedTime reaches finishThreshold (<=1).
    /// If we never enter the state, it fails gracefully without deadlock.
    /// </summary>
    private IEnumerator WaitUntilShortHashFinishes(int shortHash, float finishThreshold)
    {
        if (!animator) yield break;

        // Wait until actually in the state (brief)
        float tEnd = Time.time + 2.0f;
        while (Time.time < tEnd)
        {
            var cur = animator.GetCurrentAnimatorStateInfo(animatorLayer);
            if (cur.shortNameHash == shortHash) break;
            yield return null;
        }

        if (animator.GetCurrentAnimatorStateInfo(animatorLayer).shortNameHash != shortHash)
            yield break;

        while (true)
        {
            var cur = animator.GetCurrentAnimatorStateInfo(animatorLayer);
            if (cur.shortNameHash != shortHash) break; // left early
            if (cur.normalizedTime >= finishThreshold - 1e-4f) break;
            yield return null;
        }
    }

    /// <summary>
    /// Wait until we're in Walking (not transitioning). If timeout happens, we still proceed (non-blocking).
    /// (This prevents "freeze forever" if names are wrong.)
    /// </summary>
    private IEnumerator WaitUntilInWalkingOrTimeout()
    {
        if (!animator) yield break;

        float tEnd = Time.time + Mathf.Max(0.1f, enterWalkingTimeout);
        while (Time.time < tEnd)
        {
            bool inTrans = animator.IsInTransition(animatorLayer);
            var cur = animator.GetCurrentAnimatorStateInfo(animatorLayer);
            if (!inTrans && cur.shortNameHash == walkingHash)
                yield break;

            yield return null;
        }
    }

    // ---------------- Nav / corridor ----------------

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

    void OnDrawGizmos()
    {
        if (!drawGizmos) return;

        Gizmos.color = gizmoColor;

        Vector3 a = (Application.isPlaying || !autoDetectCorridor)
            ? endA
            : (endAOverride ? endAOverride.position : transform.position);

        Vector3 b = (Application.isPlaying || !autoDetectCorridor)
            ? endB
            : (endBOverride ? endBOverride.position : transform.position);

        Gizmos.DrawSphere(a, gizmoSphereRadius);
        Gizmos.DrawSphere(b, gizmoSphereRadius);
        Gizmos.DrawLine(a, b);
    }
}
