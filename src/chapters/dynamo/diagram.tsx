/**
 * Маленький набор для SVG-схем глав 18-19: узлы-прямоугольники, группы, рёбра-ломаные со стрелками,
 * подписи с «ореолом» и пакет, который едет по ребру. Цвета только из токенов, текст цветом --ink.
 * Схема рисуется в своих координатах (viewBox) и на узких экранах прокручивается внутри контейнера.
 */
import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import { motion, useReducedMotion } from "motion/react";
import { ArrowsHorizontalIcon } from "@phosphor-icons/react";
import { cn, cssVar, useElementSize } from "@/ui";

export type Pt = [number, number];

export interface DNode {
  id: string;
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  /** node: кликабельный блок; group: рамка-контейнер с подписью; pill: маленькая метка на ребре. */
  variant?: "node" | "group" | "pill";
  /** Цвет сущности (prefill, decode, tool…): метка-квадрат у заголовка. */
  role?: string;
  /** Можно ли кликнуть (по умолчанию да для node и pill). */
  interactive?: boolean;
  /** Заголовок сверху (место под мини-визуализацию внутри узла) или по центру (по умолчанию). */
  align?: "center" | "top";
}

export interface DEdge {
  id: string;
  points: Pt[];
  dashed?: boolean;
  label?: string;
  /** Где поставить подпись (по умолчанию середина самого длинного отрезка). */
  labelAt?: Pt;
  labelAnchor?: "start" | "middle" | "end";
  /** Кегль подписи (по умолчанию 11.5). Подпись можно разбить на строки через «\n». */
  labelSize?: number;
  /** Узлы, при выборе которых ребро подсвечивается. */
  touches?: string[];
}

export interface DiagramProps {
  width: number;
  height: number;
  nodes: DNode[];
  edges: DEdge[];
  /** Подсвеченные узлы и рёбра (шаг плеера). */
  activeNodes?: ReadonlySet<string>;
  activeEdges?: ReadonlySet<string>;
  /** Выбранный кликом узел. */
  selected?: string | null;
  onSelect?: (id: string) => void;
  /** Пакет на ребре: едет от начала к концу ломаной при смене key. */
  packet?: {
    edgeId: string;
    key: string | number;
    color?: string;
    reverse?: boolean;
  } | null;
  "aria-label": string;
  /** Минимальная ширина в пикселях: уже неё схема прокручивается. */
  minWidth?: number;
  className?: string;
  children?: ReactNode;
}

function polyline(points: Pt[]): string {
  return points.map(([x, y], i) => `${i ? "L" : "M"}${x},${y}`).join(" ");
}

function segLengths(points: Pt[]): number[] {
  const out: number[] = [];
  for (let i = 1; i < points.length; i++)
    out.push(
      Math.hypot(
        points[i][0] - points[i - 1][0],
        points[i][1] - points[i - 1][1],
      ),
    );
  return out;
}

function labelPoint(e: DEdge): Pt {
  if (e.labelAt) return e.labelAt;
  const L = segLengths(e.points);
  let k = 0;
  for (let i = 1; i < L.length; i++) if (L[i] > L[k]) k = i;
  const [a, b] = [e.points[k], e.points[k + 1]];
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 7];
}

/** Разбить подзаголовок узла на строки по ширине (грубо, по числу символов). */
function wrap(text: string, maxChars: number): string[] {
  const words = text.split(" ");
  const lines: string[] = [];
  let cur = "";
  for (const w of words) {
    if (!cur) cur = w;
    else if ((cur + " " + w).length <= maxChars) cur += " " + w;
    else {
      lines.push(cur);
      cur = w;
    }
  }
  if (cur) lines.push(cur);
  return lines;
}

export function Diagram({
  width,
  height,
  nodes,
  edges,
  activeNodes,
  activeEdges,
  selected,
  onSelect,
  packet,
  "aria-label": ariaLabel,
  minWidth = 720,
  className,
  children,
}: DiagramProps) {
  const uid = useId().replace(/:/g, "");
  const reduce = useReducedMotion() ?? false;
  const [focused, setFocused] = useState<string | null>(null);
  const markerId = `${uid}-arrow`;
  const markerActiveId = `${uid}-arrow-a`;
  const scroller = useRef<HTMLDivElement | null>(null);
  const [sizeRef, size] = useElementSize<HTMLDivElement>();
  const setScroller = useCallback(
    (el: HTMLDivElement | null) => {
      scroller.current = el;
      sizeRef(el);
    },
    [sizeRef],
  );
  const scrollable = size.width > 0 && size.width < minWidth - 1;

  // На узком экране схема прокручивается: подводим к активным узлам шага, если они за краем.
  const activeKey = activeNodes ? [...activeNodes].sort().join(",") : "";
  useEffect(() => {
    const el = scroller.current;
    if (!el || !activeKey || el.scrollWidth <= el.clientWidth + 2) return;
    const act = nodes.filter(
      (n) => n.variant !== "group" && activeNodes?.has(n.id),
    );
    if (act.length === 0) return;
    const k = el.scrollWidth / width;
    const left = Math.min(...act.map((n) => n.x)) * k;
    const right = Math.max(...act.map((n) => n.x + n.w)) * k;
    if (left >= el.scrollLeft && right <= el.scrollLeft + el.clientWidth)
      return;
    const target = Math.max(
      0,
      Math.min(
        el.scrollWidth - el.clientWidth,
        (left + right) / 2 - el.clientWidth / 2,
      ),
    );
    el.scrollTo({ left: target, behavior: reduce ? "auto" : "smooth" });
  }, [activeKey, width]);

  // в режиме шагов (есть activeEdges) подсвечиваем только рёбра шага, иначе рёбра выбранного узла
  const stepMode = Boolean(activeEdges && activeEdges.size > 0);
  const hot = (e: DEdge) =>
    stepMode
      ? activeEdges!.has(e.id)
      : Boolean(selected && e.touches?.includes(selected));
  const packetEdge = packet
    ? edges.find((e) => e.id === packet.edgeId)
    : undefined;

  const onKey = (id: string) => (ev: KeyboardEvent<SVGGElement>) => {
    if (ev.key === "Enter" || ev.key === " ") {
      ev.preventDefault();
      onSelect?.(id);
    }
  };

  return (
    <div className={cn("grid min-w-0 gap-1.5", className)}>
      <div
        ref={setScroller}
        className="min-w-0 overflow-x-auto overflow-y-hidden pb-1"
      >
        <svg
          viewBox={`0 0 ${width} ${height}`}
          role="group"
          aria-label={ariaLabel}
          className="block h-auto w-full select-none"
          style={{ minWidth }}
        >
          <defs>
            <marker
              id={markerId}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
            </marker>
            <marker
              id={markerActiveId}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="7"
              markerHeight="7"
              orient="auto-start-reverse"
            >
              <path d="M0,0 L10,5 L0,10 z" fill="var(--accent)" />
            </marker>
          </defs>

          {/* группы */}
          {nodes
            .filter((n) => n.variant === "group")
            .map((n) => (
              <g key={n.id}>
                <rect
                  x={n.x}
                  y={n.y}
                  width={n.w}
                  height={n.h}
                  rx={14}
                  fill="var(--surface-2)"
                  stroke="var(--line)"
                />
                <text
                  x={n.x + 14}
                  y={n.y + 20}
                  fontSize={12}
                  fill="var(--ink-2)"
                  fontWeight={600}
                >
                  {n.title}
                </text>
                {n.sub && (
                  <text
                    x={n.x + n.w - 14}
                    y={n.y + 20}
                    fontSize={11.5}
                    fill="var(--ink-3)"
                    textAnchor="end"
                  >
                    {n.sub}
                  </text>
                )}
              </g>
            ))}

          {/* рёбра */}
          {edges.map((e) => {
            const on = hot(e);
            return (
              <path
                key={e.id}
                d={polyline(e.points)}
                fill="none"
                stroke={on ? "var(--accent)" : "var(--ink-3)"}
                strokeWidth={on ? 2 : 1.4}
                strokeDasharray={e.dashed ? "5 4" : undefined}
                strokeLinejoin="round"
                markerEnd={`url(#${on ? markerActiveId : markerId})`}
                opacity={activeEdges && activeEdges.size > 0 && !on ? 0.55 : 1}
              />
            );
          })}

          {/* узлы */}
          {nodes
            .filter((n) => n.variant !== "group")
            .map((n) => {
              const pill = n.variant === "pill";
              const active = activeNodes?.has(n.id) ?? false;
              const isSel = selected === n.id;
              const interactive = n.interactive ?? Boolean(onSelect);
              // приглушаем цветом, а не прозрачностью группы: подписи неактивных узлов остаются не ниже AA
              const dim = Boolean(activeNodes && activeNodes.size > 0 && !active && !isSel);
              const subLines = n.sub
                ? wrap(n.sub, Math.max(8, Math.floor((n.w - 20) / 6.4)))
                : [];
              const titleY = pill
                ? n.y + n.h / 2 + 4.5
                : n.align === "top"
                  ? n.y + 24
                  : n.y +
                    (n.h - (subLines.length ? 16 + subLines.length * 15 : 0)) /
                      2 +
                    (subLines.length ? 13 : 5);
              return (
                <g
                  key={n.id}
                  role={interactive ? "button" : undefined}
                  tabIndex={interactive ? 0 : undefined}
                  aria-pressed={interactive ? isSel : undefined}
                  aria-label={
                    interactive
                      ? `${n.title}${n.sub ? `: ${n.sub}` : ""}`
                      : undefined
                  }
                  onClick={interactive ? () => onSelect?.(n.id) : undefined}
                  onKeyDown={interactive ? onKey(n.id) : undefined}
                  onFocus={(e) => {
                    // кольцо фокуса только для клавиатуры: после клика мышью хватает выделения узла
                    if (e.currentTarget.matches(":focus-visible")) setFocused(n.id);
                  }}
                  onBlur={() => setFocused((f) => (f === n.id ? null : f))}
                  className={cn(
                    interactive && "cursor-pointer",
                    "outline-none",
                  )}
                >
                  <rect
                    x={n.x}
                    y={n.y}
                    width={n.w}
                    height={n.h}
                    rx={pill ? n.h / 2 : 10}
                    fill={active ? "var(--accent-soft)" : "var(--surface)"}
                    stroke={
                      isSel || active
                        ? "var(--accent)"
                        : dim
                          ? "var(--line)"
                          : "var(--line-strong)"
                    }
                    strokeWidth={isSel ? 2 : active ? 1.6 : 1}
                    strokeDasharray={pill ? "4 3" : undefined}
                  />
                  {isSel && interactive && !pill && (
                    // выбранный кликом узел: акцентная «вкладка» на верхней кромке, чтобы не путать с активным шагом
                    <rect
                      x={n.x + 12}
                      y={n.y - 3}
                      width={30}
                      height={6}
                      rx={3}
                      fill="var(--accent)"
                    />
                  )}
                  {focused === n.id && (
                    <rect
                      x={n.x - 4}
                      y={n.y - 4}
                      width={n.w + 8}
                      height={n.h + 8}
                      rx={pill ? (n.h + 8) / 2 : 13}
                      fill="none"
                      stroke="var(--accent)"
                      strokeWidth={2}
                    />
                  )}
                  {n.role && !pill && (
                    <rect
                      x={n.x + 12}
                      y={titleY - 10}
                      width={10}
                      height={10}
                      rx={2.5}
                      fill={cssVar(n.role as never)}
                      opacity={dim ? 0.45 : 1}
                    />
                  )}
                  <text
                    x={pill ? n.x + n.w / 2 : n.x + (n.role ? 28 : 12)}
                    y={titleY}
                    fontSize={pill ? 12 : 13.5}
                    fontWeight={pill ? 500 : 600}
                    fill={dim ? "var(--ink-2)" : "var(--ink)"}
                    textAnchor={pill ? "middle" : "start"}
                  >
                    {n.title}
                  </text>
                  {subLines.map((line, i) => (
                    <text
                      key={i}
                      x={n.x + 12}
                      y={titleY + 18 + i * 15}
                      fontSize={12}
                      fill={dim ? "var(--ink-3)" : "var(--ink-2)"}
                    >
                      {line}
                    </text>
                  ))}
                </g>
              );
            })}

          {/* подписи рёбер поверх узлов не нужны: рисуем после рёбер, с ореолом */}
          {edges
            .filter((e) => e.label)
            .map((e) => {
              const [x, y] = labelPoint(e);
              const on = hot(e);
              const size = e.labelSize ?? 11.5;
              const lines = e.label!.split("\n");
              return (
                <text
                  key={`l-${e.id}`}
                  x={x}
                  y={y}
                  fontSize={size}
                  textAnchor={e.labelAnchor ?? "middle"}
                  fill={on ? "var(--ink)" : "var(--ink-2)"}
                  fontWeight={on ? 600 : 400}
                  stroke="var(--surface)"
                  strokeWidth={4}
                  strokeLinejoin="round"
                  paintOrder="stroke"
                >
                  {lines.length === 1
                    ? e.label
                    : lines.map((line, i) => (
                        <tspan key={i} x={x} dy={i === 0 ? 0 : size * 1.15}>
                          {line}
                        </tspan>
                      ))}
                </text>
              );
            })}

          {packetEdge && packet && (
            <Packet
              key={`${packet.edgeId}-${packet.key}`}
              edge={packetEdge}
              color={packet.color}
              reduce={reduce}
              reverse={packet.reverse}
            />
          )}
          {children}
        </svg>
      </div>
      {scrollable && (
        <div className="flex items-center gap-1.5 text-[12.5px] text-ink-3">
          <ArrowsHorizontalIcon size={16} aria-hidden />
          схема шире экрана: прокрутите её вбок
        </div>
      )}
    </div>
  );
}

function Packet({
  edge,
  color = "prefill",
  reduce,
  reverse,
}: {
  edge: DEdge;
  color?: string;
  reduce: boolean;
  reverse?: boolean;
}) {
  const pts = reverse ? edge.points.slice().reverse() : edge.points;
  const L = segLengths(pts);
  const total = L.reduce((a, b) => a + b, 0) || 1;
  const times = [0];
  let acc = 0;
  for (const l of L) {
    acc += l;
    times.push(acc / total);
  }
  const fill = color.startsWith("var(") ? color : cssVar(color as never);
  const end = pts[pts.length - 1];
  if (reduce)
    return (
      <circle
        cx={end[0]}
        cy={end[1]}
        r={6}
        fill={fill}
        stroke="var(--surface)"
        strokeWidth={2}
      />
    );
  return (
    <motion.circle
      r={6}
      fill={fill}
      stroke="var(--surface)"
      strokeWidth={2}
      initial={{ cx: pts[0][0], cy: pts[0][1], opacity: 0 }}
      animate={{
        cx: pts.map((p) => p[0]),
        cy: pts.map((p) => p[1]),
        opacity: 1,
      }}
      transition={{
        default: {
          duration: Math.min(1.4, 0.35 + total / 700),
          times,
          ease: "easeInOut",
        },
        opacity: { duration: 0.15 },
      }}
    />
  );
}

/** Подложка-«пятно» цвета сущности для легенд над схемами. */
export function Swatch({ role, label }: { role: string; label: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[13px] text-ink-2">
      <span
        aria-hidden
        className="h-2.5 w-2.5 rounded-[3px]"
        style={{ background: cssVar(role as never) }}
      />
      {label}
    </span>
  );
}
