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

/**
 * The two things stripped out of the prose: a citation marker, and the `**`
 * the synthesis prompt uses to label a figure ("**Fiscal Year 2025:**").
 *
 * Deliberately not a markdown parser — see renderAnswer's note. `**` is the
 * one delimiter the prompt actually produces, it cannot occur inside a dollar
 * figure, and it is doubled, so it cannot collide with the `*` that starts a
 * bullet. Underscores are pointedly NOT emphasis here: metric names arrive as
 * gross_margin and net_income, and italicising half of one would be a
 * renderer inventing formatting out of data.
 */
const MARKER_RE = /\[SRC:([A-Z_]+):([^\]]+)\]|\*\*/g;

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
function extract(block: string): { text: string; markers: Marker[]; bolds: [number, number][] } {
  let text = '';
  const markers: Marker[] = [];
  const bolds: [number, number][] = [];
  let last = 0;
  let boldFrom = -1;

  MARKER_RE.lastIndex = 0;
  for (let match = MARKER_RE.exec(block); match !== null; match = MARKER_RE.exec(block)) {
    text += block.slice(last, match.index);
    last = match.index + match[0].length;

    if (match[1] !== undefined) {
      markers.push({ at: text.length, key: `${match[1]}:${match[2]}`.toUpperCase() });
    } else if (boldFrom === -1) {
      boldFrom = text.length;
    } else {
      bolds.push([boldFrom, text.length]);
      boldFrom = -1;
    }
  }
  text += block.slice(last);

  // An unclosed `**` is a model artefact rather than emphasis. The delimiter
  // is already gone from `text`; dropping the range with it means the prose
  // reads normally instead of turning bold to the end of the paragraph.
  return { text, markers, bolds };
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
  | { at: number; kind: 'close' }
  | { at: number; kind: 'bold-open' }
  | { at: number; kind: 'bold-close' };

/**
 * Order of events landing on the same offset.
 *
 * Everything closes before anything opens, and a citation superscript sits
 * outside the emphasis it follows — "**FY2025:**[SRC:...]" should render the
 * reference after the bold run, not inside it.
 */
const RANK: Record<Event['kind'], number> = {
  close: 0,
  'bold-close': 1,
  cite: 2,
  'bold-open': 3,
  open: 4,
};

/**
 * Render one paragraph: prose, superscript references, and highlighted spans
 * for anything the verifier could not ground.
 */
function renderBlock(block: string, index: Map<string, number>, claims: string[], keyPrefix: string): ReactNode[] {
  const { text, markers, bolds } = extract(block);

  const spans = claims.flatMap((claim) => locate(text, claim));
  const events: Event[] = [
    ...markers.map((m): Event => ({ at: m.at, kind: 'cite', key: m.key })),
    ...spans.map(([start]): Event => ({ at: start, kind: 'open' })),
    ...spans.map(([, end]): Event => ({ at: end, kind: 'close' })),
    ...bolds.map(([start]): Event => ({ at: start, kind: 'bold-open' })),
    ...bolds.map(([, end]): Event => ({ at: end, kind: 'bold-close' })),
  ].sort((a, b) => a.at - b.at || RANK[a.kind] - RANK[b.kind]);

  const out: ReactNode[] = [];
  let cursor = 0;
  let open = false;
  let bold = false;
  let seq = 0;
  let buffer: ReactNode[] = [];

  // Emphasis wraps the text slice rather than switching the flush wrapper, so
  // it composes with a highlight instead of fighting it: a flagged claim that
  // happens to contain a bold label still renders as one <mark>.
  const pushText = (slice: string) => {
    if (!slice) return;
    buffer.push(bold ? <strong key={`${keyPrefix}-s${seq++}`}>{slice}</strong> : slice);
  };

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
      pushText(text.slice(cursor, event.at));
      cursor = event.at;
    }
    if (event.kind === 'bold-open') {
      bold = true;
    } else if (event.kind === 'bold-close') {
      bold = false;
    } else if (event.kind === 'cite') {
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

  if (cursor < text.length) pushText(text.slice(cursor));
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
 * The synthesis prompt produces paragraphs separated by blank lines,
 * markdown-ish `*   ` bullets, and `**` around the label it puts in front of
 * a figure. Those three, and nothing else: an answer is prose with
 * references, and a full markdown pipeline would invite the renderer to read
 * a dollar figure or the underscore in gross_margin as formatting.
 *
 * `**` is handled rather than left as literal asterisks because the model
 * emits it on essentially every multi-period answer, and unrendered
 * delimiters in the middle of financial prose read as a broken product.
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
