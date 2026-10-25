import { useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { pageKey, type FlightEvidence, type FlightLocation } from "./flight";

type Point = { x: number; y: number };

export function FlightOverlay({ sceneRef, location, evidence, delivering, complete, supportingIds, scanning, evidenceLabel, routeKey, layoutKey, paused, parked, holding, duration, calm }: {
  sceneRef: RefObject<HTMLDivElement>;
  location: FlightLocation | null;
  evidence: FlightEvidence | null;
  delivering: boolean;
  complete: boolean;
  supportingIds: string[];
  scanning: boolean;
  evidenceLabel: string;
  routeKey: string;
  layoutKey: string;
  paused: boolean;
  parked: boolean;
  holding: boolean;
  duration: number;
  calm: boolean;
}) {
  const pathRef = useRef<SVGPathElement>(null);
  const planeRef = useRef<SVGGElement>(null);
  const hullRef = useRef<SVGGElement>(null);
  const labelRef = useRef<SVGGElement>(null);
  const beamRef = useRef<SVGPathElement>(null);
  const trailRef = useRef<SVGPathElement>(null);
  const pointRef = useRef<Point | null>(null);
  const elapsedRef = useRef(0);
  const [flying, setFlying] = useState(false);
  const [route, setRoute] = useState<{ path: string; start: Point; end: Point; source: Point | null; width: number; height: number } | null>(null);
  const [connections, setConnections] = useState<string[]>([]);

  useLayoutEffect(() => {
    const scene = sceneRef.current;
    if (!scene) return;
    function updateRoute() {
      if (!scene) return;
      const bounds = scene.getBoundingClientRect();
      const dock = scene.querySelector<HTMLElement>(".flight-takeoff-point");
      const paper = !parked && location ? Array.from(scene.querySelectorAll<HTMLElement>("[data-flight-page]"))
        .find((element) => element.dataset.flightPage === pageKey(location)) : null;
      const target = paper ?? dock;
      if (!target) return;
      const targetBounds = target.getBoundingClientRect();
      const anchor = paper && evidence ? Array.from(paper.querySelectorAll<HTMLElement>("[data-flight-anchor]"))
        .find((element) => element.dataset.flightAnchor === evidence.id)?.getBoundingClientRect() : null;
      const box = paper ? evidence?.box : null;
      const source = paper ? {
        x: anchor ? anchor.left - bounds.left : targetBounds.left - bounds.left + targetBounds.width * (box ? (box.left + box.width / 2) / 100 : 0.5),
        y: anchor ? anchor.top - bounds.top : targetBounds.top - bounds.top + targetBounds.height * (box ? (box.top + box.height / 2) / 100 : 0.5),
      } : null;
      const pageBounds = paper?.closest(".flight-page-viewport")?.getBoundingClientRect();
      const bookBounds = paper?.closest(".flight-bound-book")?.getBoundingClientRect();
      const destination = pageBounds ? {
        x: pageBounds.left - bounds.left + pageBounds.width * 0.82,
        y: (bookBounds?.top ?? pageBounds.top) - bounds.top + 30,
      } : {
        x: targetBounds.left - bounds.left + targetBounds.width * (box ? (box.left + box.width / 2) / 100 : 0.5),
        y: targetBounds.top - bounds.top + targetBounds.height / 2,
      };
      const clampPoint = (point: Point) => ({ x: Math.max(80, Math.min(bounds.width - 80, point.x)), y: Math.max(95, Math.min(bounds.height - 85, point.y)) });
      const end = clampPoint(destination);
      const start = clampPoint(pointRef.current ?? end);
      const lift = Math.min(56, Math.hypot(end.x - start.x, end.y - start.y) * 0.18);
      elapsedRef.current = 0;
      setRoute({ path: `M ${start.x} ${start.y} Q ${(start.x + end.x) / 2} ${Math.max(75, Math.min(start.y, end.y) - lift)} ${end.x} ${end.y}`, start, end, source, width: bounds.width, height: bounds.height });
      const answer = scene.querySelector<HTMLElement>(".flight-answer-heading")?.getBoundingClientRect();
      const list = scene.querySelector<HTMLElement>(".flight-evidence-list")?.getBoundingClientRect();
      setConnections(complete && answer && list ? Array.from(scene.querySelectorAll<HTMLElement>("[data-flight-evidence]"))
        .filter((element) => supportingIds.includes(element.dataset.flightEvidence ?? ""))
        .flatMap((element) => {
          const card = element.getBoundingClientRect();
          if (card.bottom <= list.top || card.top >= list.bottom) return [];
          const startX = card.left - bounds.left - 6;
          const startY = Math.max(card.top + 12, list.top + 12) - bounds.top;
          const endX = answer.left - bounds.left - 3;
          const endY = answer.top - bounds.top + 10;
          return [`M ${startX} ${startY} C ${startX - 19} ${startY} ${endX - 19} ${endY} ${endX} ${endY}`];
        }) : []);
    }
    updateRoute();
    const observer = new ResizeObserver(updateRoute);
    observer.observe(scene);
    const book = scene.querySelector(".flight-bound-book");
    if (book) observer.observe(book);
    scene.querySelectorAll(".flight-source-page").forEach((page) => observer.observe(page));
    function handleScroll(event: Event) {
      const results = scene?.querySelector(".flight-results");
      if ((event.target instanceof Element && event.target.classList.contains("flight-page-viewport")) || (results && (event.target === results || (complete && event.target instanceof Node && results.contains(event.target))))) updateRoute();
    }
    scene.addEventListener("scroll", handleScroll, true);
    return () => { observer.disconnect(); scene.removeEventListener("scroll", handleScroll, true); };
  }, [sceneRef, location?.docId, location?.page, evidence, delivering, complete, supportingIds, scanning, parked, routeKey, layoutKey]);

  useEffect(() => {
    const path = pathRef.current;
    const plane = planeRef.current;
    if (!route || !path || !plane) return;
    function positionAccessories(position: Point, progress?: number) {
      if (!route) return;
      labelRef.current?.setAttribute("transform", `translate(${Math.max(8, Math.min(route.width - 134, position.x - 62))} ${Math.max(7, position.y - 43)})`);
      if (beamRef.current && route.source) {
        beamRef.current.setAttribute("d", `M ${position.x} ${position.y + 18} L ${route.source.x - 28} ${route.source.y + 8} Q ${route.source.x} ${route.source.y + 22} ${route.source.x + 28} ${route.source.y + 8} Z`);
        if (progress !== undefined) beamRef.current.style.opacity = progress > 0.88 ? "1" : "0";
      }
    }
    if (parked || calm || Math.hypot(route.end.x - route.start.x, route.end.y - route.start.y) < 1 || elapsedRef.current >= duration) {
      setFlying(false);
      pointRef.current = route.end;
      plane.setAttribute("transform", `translate(${route.end.x} ${route.end.y})`);
      hullRef.current?.removeAttribute("transform");
      positionAccessories(route.end, 1);
      trailRef.current?.setAttribute("stroke-dasharray", "0 10000");
      return;
    }
    if (paused) { setFlying(false); positionAccessories(pointRef.current ?? route.start); return; }
    setFlying(true);
    let frame = 0;
    let previous = performance.now();
    const length = path.getTotalLength();
    function animate(now: number) {
      if (!path || !plane || !route) return;
      elapsedRef.current += now - previous;
      previous = now;
      const progress = Math.min(1, elapsedRef.current / duration);
      const eased = progress * progress * (3 - 2 * progress);
      const position = path.getPointAtLength(length * eased);
      const before = path.getPointAtLength(Math.max(0, length * eased - 2));
      const next = path.getPointAtLength(Math.min(length, length * eased + 2));
      const angle = Math.atan2(next.y - before.y, next.x - before.x) * 180 / Math.PI;
      const bank = Math.sin(progress * Math.PI * 2) * 8;
      pointRef.current = position;
      plane.setAttribute("transform", `translate(${position.x} ${position.y}) rotate(${angle})`);
      hullRef.current?.setAttribute("transform", `skewX(${bank}) scale(1 ${1 - Math.abs(bank) * 0.006})`);
      trailRef.current?.setAttribute("stroke-dasharray", `${length * eased} ${length}`);
      positionAccessories(position, progress);
      if (progress < 1) frame = requestAnimationFrame(animate);
      else {
        setFlying(false);
        plane.setAttribute("transform", `translate(${route.end.x} ${route.end.y})`);
        hullRef.current?.removeAttribute("transform");
      }
    }
    frame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frame);
  }, [route, duration, paused, parked, calm, scanning, delivering]);

  const pilotLabel = complete ? "MISSION COMPLETE" : parked ? "PILOT READY" : delivering ? `RECORDED ${evidenceLabel}` : scanning ? `SCANNING / PAGE ${location?.page}` : holding ? location ? `WAITING / PAGE ${location.page}` : "WAITING FOR A PAGE" : `EXPLORING / PAGE ${location?.page ?? "—"}`;
  return <svg className={`flight-overlay ${calm ? "calm" : ""} ${scanning ? "scanning" : ""} ${parked ? "parked" : ""} ${flying ? "flying" : "stationary"}`} aria-hidden="true">
    <defs>
      <linearGradient id="flight-paper-top" x1="0" y1="0" x2="0.7" y2="1"><stop stopColor="#fffef6" /><stop offset="0.55" stopColor="#f7e9cc" /><stop offset="1" stopColor="#dfc18a" /></linearGradient>
      <linearGradient id="flight-paper-fold" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#cceef0" /><stop offset="1" stopColor="#5794ac" /></linearGradient>
      <linearGradient id="flight-searchlight" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#6be5f155" /><stop offset="1" stopColor="#72e9c80a" /></linearGradient>
      {route ? <linearGradient id="flight-route-gradient" gradientUnits="userSpaceOnUse" x1={route.start.x} y1={route.start.y} x2={route.end.x} y2={route.end.y}><stop stopColor="#67dded00" /><stop offset="1" stopColor="#7de7f1" /></linearGradient> : null}
    </defs>
    {connections.map((connection, index) => <path className="flight-answer-connection" key={index} d={connection} style={{ animationDelay: `${index * 120}ms` }} />)}
    {route ? <>
      <path className="flight-route-glow" d={route.path} />
      <path ref={pathRef} className="flight-route" d={route.path} style={{ animationPlayState: paused ? "paused" : "running" }} />
      <path ref={trailRef} className="flight-route-trail" d={route.path} strokeDasharray="0 10000" />
      {scanning && route.source ? <>
        <path ref={beamRef} className="flight-searchlight" />
        <g className="flight-source-radar" transform={`translate(${route.source.x} ${route.source.y})`}><circle r="25" /><circle r="40" /><circle className="flight-source-pulse" r="24" /><path d="M-40-24v-16h16m48 0h16v16m0 48v16H24m-48 0h-16V24M-7 0H7M0-7V7" /></g>
      </> : null}
      <g ref={planeRef} className="flight-airplane" transform={`translate(${route.start.x} ${route.start.y})`}>
          <g className="flight-plane-size">
            <g className="flight-paper-wake"><path d="M-23-5Q-34-10-46-6" /><path d="M-23 5Q-37 10-50 6" /></g>
            <g ref={hullRef} className="flight-plane-hull">
              <ellipse className="flight-plane-shadow" cx="-2" cy="5" rx="19" ry="9" />
              <path className="flight-plane-body" d="M24 0-20-16-12 0-20 16Z" />
              <path className="flight-plane-facet" d="M24 0-12 0-20 16-5 8Z" />
              <path className="flight-plane-fold" d="M24 0-12 0M24 0-5 8-9 4" />
              <path className="flight-paper-glint" d="M-17-13 18-1" />
            </g>
          </g>
      </g>
      <g ref={labelRef} className="flight-pilot-label" transform={`translate(${route.start.x - 62} ${Math.max(7, route.start.y - 43)})`}><rect width="124" height="20" rx="10" /><circle cx="11" cy="10" r="2" /><text x="20" y="13">{pilotLabel}</text></g>
    </> : null}
  </svg>;
}
