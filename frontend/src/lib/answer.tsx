// ═══════════════════════════════════════════════════════
// FinSight web — answer rendering
// ═══════════════════════════════════════════════════════
//
// Purpose : Turn the drafted answer into prose a person can read, without
//           throwing away the two things that make FinSight's answers worth
//           more than a chatbot's — where each number came from, and which
//           claims could not be traced.
//
// ══ WHAT THE BACKEND ACTUALLY SENDS ══
//   Inline markers in the exact form [SRC:TYPE:ID], placed immediately after
//   the claim they support (src/research/config.py). The Streamlit dashboard
//   rendered them raw, so answers read as prose with machine noise stapled
//   through them. Here each marker becomes a superscript reference into the
//   source list underneath.
//
// ══ WHY FLAGGED CLAIMS ARE MARKED AND NEVER REMOVED ══
//   The verifier's failure mode is dropping claims that are TRUE but phrased
//   in a way it could not ground. Deleting on its say-so would silently lose
//   correct analysis; highlighting hands the judgement to the person reading,
//   which is also what the design brief asks for. A claim that cannot be
//   located in the prose still gets listed under the answer — it is never
//   dropped just because the highlighter missed it.
// ═══════════════════════════════════════════════════════

import type { ReactNode } from 'react';
import type { Citation, UnsupportedClaim } from '../api/types';

const MARKER_RE = /\[SRC:([A-Z_]+):([^\]]+)\]/g;

/** `TYPE:ID` -> its 1-based position in the source list. */
export function citationIndex(citations: Citation[]): Map<string, number> {
  const index = new Map<string, number>();
  citations.forEach((cite, i) => {
    const key = `${cite.source_type}:${cite.source_id}`.toUpperCase();
    if (!index.has(key)) index.set(key, i + 1);
  });
  return index;
}

interface Marker {
  /** Offset into the marker-free text. */
  at: number;
  key: string;
}

/** Strip the inline markers out of one block, remembering where they were. */
function extract(block: string): { text: string; markers: Marker[] } {
  let text = '';
  const markers: Marker[] = [];
  let last = 0;

  MARKER_RE.lastIndex = 0;
  for (let match = MARKER_RE.exec(block); match !== null; match = MARKER_RE.exec(block)) {
    text += block.slice(last, match.index);
    markers.push({ at: text.length, key: `${match[1]}:${match[2]}`.toUpperCase() });
    last = match.index + match[0].length;
  }
  text += block.slice(last);

  return { text, markers };
}

/**
 * Where an unsupported claim sits in the marker-free text.
 *
 * Whitespace-insensitive, because the claim the verifier reports has usually
 * been through a model and may differ from the prose by a line break. Returns
 * [] when the claim cannot be located — the caller lists it separately rather
 * than guessing.
 */
function locate(text: string, claim: string): [number, number][] {
  const needle = claim.trim().replace(/\s+/g, ' ');
  if (needle.length < 8) return [];

  const haystack = text.replace(/\s+/g, ' ');
  const at = haystack.toLowerCase().indexOf(needle.toLowerCase());
  if (at === -1) return [];

  // Map the collapsed-whitespace offsets back onto the original text, which
  // may contain runs the collapsed copy turned into single spaces.
  let original = 0;
  let collapsed = 0;
  let start = -1;
  let end = -1;
  while (original < text.length) {
    if (collapsed === at && start === -1) start = original;
    if (collapsed === at + needle.length) {
      end = original;
      break;
    }
    const isSpace = /\s/.test(text[original]);
    if (isSpace) {
      while (original < text.length && /\s/.test(text[original])) original++;
      collapsed++;
    } else {
      original++;
      collapsed++;
    }
  }
  if (start === -1) return [];
  return [[start, end === -1 ? text.length : end]];
}

type Event =
  | { at: number; kind: 'cite'; key: string }
  | { at: number; kind: 'open' }
  | { at: number; kind: 'close' };

/**
 * Render one paragraph: prose, superscript references, and highlighted spans
 * for anything the verifier could not ground.
 */
function renderBlock(block: string, index: Map<string, number>, claims: string[], keyPrefix: string): ReactNode[] {
  const { text, markers } = extract(block);

  const spans = claims.flatMap((claim) => locate(text, claim));
  const events: Event[] = [
    ...markers.map((m): Event => ({ at: m.at, kind: 'cite', key: m.key })),
    ...spans.map(([start]): Event => ({ at: start, kind: 'open' })),
    ...spans.map(([, end]): Event => ({ at: end, kind: 'close' })),
  ].sort((a, b) => a.at - b.at || (a.kind === 'close' ? -1 : 1));

  const out: ReactNode[] = [];
  let cursor = 0;
  let open = false;
  let buffer: ReactNode[] = [];

  const flush = () => {
    if (buffer.length === 0) return;
    if (open) {
      out.push(
        <mark
          key={`${keyPrefix}-m${out.length}`}
          style={{
            background: 'color-mix(in srgb, var(--alert) 22%, transparent)',
            color: 'inherit',
            padding: '0 3px',
            borderBottom: '1px solid var(--alert)',
          }}
        >
          {buffer}
        </mark>,
      );
    } else {
      out.push(<span key={`${keyPrefix}-t${out.length}`}>{buffer}</span>);
    }
    buffer = [];
  };

  for (const event of events) {
    if (event.at > cursor) {
      buffer.push(text.slice(cursor, event.at));
      cursor = event.at;
    }
    if (event.kind === 'cite') {
      const n = index.get(event.key);
      buffer.push(
        <sup
          key={`${keyPrefix}-c${buffer.length}-${event.key}`}
          className="mono"
          title={event.key}
          style={{ fontSize: '0.66em', color: 'var(--color-accent)', padding: '0 1px', letterSpacing: '0.04em' }}
        >
          {n ? n : '·'}
        </sup>,
      );
    } else if (event.kind === 'open') {
      flush();
      open = true;
    } else {
      flush();
      open = false;
    }
  }

  if (cursor < text.length) buffer.push(text.slice(cursor));
  flush();
  return out;
}

export interface AnswerBlock {
  kind: 'paragraph' | 'bullet';
  content: ReactNode[];
}

/**
 * Split the answer into blocks and render each.
 *
 * The synthesis prompt produces paragraphs separated by blank lines and
 * markdown-ish `*   ` bullets. Nothing more elaborate is parsed: an answer is
 * prose with references, and running it through a full markdown pipeline
 * would invite the renderer to interpret a dollar figure or an underscore in
 * a ticker as formatting.
 */
export function renderAnswer(
  answer: string,
  citations: Citation[],
  unsupported: UnsupportedClaim[] = [],
): AnswerBlock[] {
  const index = citationIndex(citations);
  const claims = unsupported.map((u) => u.claim);

  return answer
    .split(/\n{2,}/)
    .flatMap((chunk) => chunk.split('\n'))
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, i) => {
      const bullet = /^[*-]\s+/.test(line);
      return {
        kind: bullet ? ('bullet' as const) : ('paragraph' as const),
        content: renderBlock(bullet ? line.replace(/^[*-]\s+/, '') : line, index, claims, `b${i}`),
      };
    });
}

/** Which flagged claims never got highlighted, so they can be listed instead. */
export function unlocatedClaims(answer: string, unsupported: UnsupportedClaim[]): UnsupportedClaim[] {
  const { text } = extract(answer);
  return unsupported.filter((u) => locate(text, u.claim).length === 0);
}
