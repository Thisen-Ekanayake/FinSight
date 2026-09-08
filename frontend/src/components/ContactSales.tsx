// ═══════════════════════════════════════════════════════
// FinSight web — the free tier's ceiling
// ═══════════════════════════════════════════════════════
//
// ══ WHY THIS IS NOT AN ErrorNote ══
//   ErrorNote's whole point is that it reports a FAILURE — a backend that is
//   unreachable, a call that did not work. Running out of free queries is
//   neither. Rendering it in the same monospace failure box would tell the
//   user that something broke, when what actually happened is that they
//   reached the end of the trial and there is a way forward.
//
// ══ THE URL IS A PROP, NEVER A CONSTANT ══
//   It comes from the server — CONTACT_URL, served in the 402 body and by
//   GET /auth/quota. Hardcoding the repo here would mean a fork or a private
//   deployment could not point it anywhere else without a rebuild, which is
//   the same reason the Google client ID is served rather than baked in.
// ═══════════════════════════════════════════════════════

import { Blueprint, Kicker } from './primitives';

export function ContactSales({
  contactUrl,
  used,
  limit,
}: {
  contactUrl: string;
  used?: number;
  limit?: number;
}) {
  return (
    <Blueprint style={{ padding: '26px 28px', maxWidth: '68ch' }}>
      <Kicker>Free tier</Kicker>

      <h2
        style={{
          fontSize: 22,
          lineHeight: 1.25,
          margin: '0 0 10px',
          fontFamily: 'var(--font-heading)',
          fontWeight: 'var(--font-heading-weight)' as never,
        }}
      >
        {typeof limit === 'number' && limit > 0
          ? `You've used all ${limit} free queries.`
          : 'This account has no free queries left.'}
      </h2>

      <p style={{ fontSize: 14, lineHeight: 1.7, color: 'var(--ink-70)', margin: '0 0 20px' }}>
        Every query runs a full multi-agent research pass — several model calls and a fan-out
        across live data sources — so the free allowance is per account and does not reset.
        Everything you have already asked stays readable under Findings.
      </p>

      <a
        className="btn btn-primary"
        href={contactUrl}
        target="_blank"
        // noreferrer alongside noopener: the target is a link the server
        // chose, and it has no business learning which page sent the user.
        rel="noreferrer noopener"
      >
        Contact sales
      </a>

      {typeof used === 'number' && typeof limit === 'number' && limit > 0 ? (
        <p className="mono" style={{ fontSize: 12, color: 'var(--ink-45)', margin: '18px 0 0' }}>
          {used} / {limit} used
        </p>
      ) : null}
    </Blueprint>
  );
}
