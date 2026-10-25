import { useEffect, useId, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";
import { createPortal } from "react-dom";

type SelectOption = { value: string; label: string; description?: string };

export function WorkspaceSelect({
  className, label, title, value, options, disabled, onChange,
}: {
  className: string;
  label: string;
  title?: string;
  value: string;
  options: SelectOption[];
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  const id = useId();
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const typeaheadRef = useRef({ text: "", time: 0 });
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const [position, setPosition] = useState<{ left: number; top: number; width: number; maxHeight: number } | null>(null);
  const selected = options.find((option) => option.value === value);
  const isOpen = open && !disabled && options.length > 0;

  function openMenu(index = Math.max(0, options.findIndex((option) => option.value === value))) {
    setActiveIndex(index);
    setPosition(null);
    setOpen(true);
  }

  function selectOption(index: number) {
    setOpen(false);
    triggerRef.current?.focus({ preventScroll: true });
    if (options[index] && options[index].value !== value) onChange(options[index].value);
  }

  useEffect(() => {
    if (disabled || options.length === 0) setOpen(false);
  }, [disabled, options.length]);

  useLayoutEffect(() => {
    if (!isOpen || !triggerRef.current || !menuRef.current) return;
    const bounds = triggerRef.current.getBoundingClientRect();
    const width = Math.min(Math.max(bounds.width, 240), window.innerWidth - 24);
    menuRef.current.style.width = `${width}px`;
    const height = Math.min(menuRef.current.scrollHeight + 2, 320);
    const below = window.innerHeight - bounds.bottom - 18;
    const above = bounds.top - 18;
    const upwards = below < height && above > below;
    const maxHeight = Math.max(0, Math.min(height, upwards ? above : below));
    setPosition({
      left: Math.max(12, Math.min(bounds.left, window.innerWidth - width - 12)),
      top: upwards ? bounds.top - maxHeight - 6 : bounds.bottom + 6,
      width,
      maxHeight,
    });
  }, [isOpen, options.length]);

  useEffect(() => {
    if (!isOpen) return;
    function handleOutside(event: PointerEvent) {
      const target = event.target as Node;
      if (!triggerRef.current?.contains(target) && !menuRef.current?.contains(target)) setOpen(false);
    }
    function handleScroll(event: Event) {
      if (event.target instanceof Node && menuRef.current?.contains(event.target)) return;
      setOpen(false);
    }
    const close = () => setOpen(false);
    document.addEventListener("pointerdown", handleOutside);
    window.addEventListener("scroll", handleScroll, true);
    window.addEventListener("resize", close);
    return () => {
      document.removeEventListener("pointerdown", handleOutside);
      window.removeEventListener("scroll", handleScroll, true);
      window.removeEventListener("resize", close);
    };
  }, [isOpen]);

  useEffect(() => {
    if (isOpen && position) menuRef.current?.children[activeIndex]?.scrollIntoView({ block: "nearest" });
  }, [isOpen, activeIndex, position]);

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "Tab" || event.key === "Escape") {
      setOpen(false);
      if (event.key === "Escape" && isOpen) event.preventDefault();
      return;
    }
    if (["ArrowDown", "ArrowUp", "Home", "End", "Enter", " "].includes(event.key)) {
      event.preventDefault();
      if (!isOpen) {
        openMenu(event.key === "Home" ? 0 : event.key === "End" ? options.length - 1 : undefined);
      } else if (event.key === "Enter" || event.key === " ") {
        selectOption(activeIndex);
      } else {
        setActiveIndex((current) => event.key === "Home" ? 0 : event.key === "End" ? options.length - 1
          : (current + (event.key === "ArrowDown" ? 1 : -1) + options.length) % options.length);
      }
      return;
    }
    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.altKey) {
      event.preventDefault();
      const now = Date.now();
      const text = (now - typeaheadRef.current.time < 700 ? typeaheadRef.current.text : "") + event.key.toLowerCase();
      typeaheadRef.current = { text, time: now };
      const index = options.findIndex((option) => option.label.toLowerCase().startsWith(text));
      if (index >= 0) {
        if (isOpen) setActiveIndex(index);
        else openMenu(index);
      }
    }
  }

  return <>
    <button
      ref={triggerRef}
      className={className}
      type="button"
      role="combobox"
      aria-label={label}
      aria-haspopup="listbox"
      aria-expanded={isOpen}
      aria-controls={isOpen ? id : undefined}
      aria-activedescendant={isOpen ? `${id}-${activeIndex}` : undefined}
      title={title}
      disabled={disabled || options.length === 0}
      onClick={() => isOpen ? setOpen(false) : openMenu()}
      onKeyDown={handleKeyDown}
      onBlur={() => setOpen(false)}
    >{selected?.label ?? "No files"}</button>
    {isOpen ? createPortal(<div
      ref={menuRef}
      id={id}
      className={`workspace-select-menu ${className}-menu`}
      role="listbox"
      aria-label={label}
      style={position ?? { visibility: "hidden" }}
      onPointerDown={(event) => event.preventDefault()}
    >{options.map((option, index) => <div
      key={option.value}
      id={`${id}-${index}`}
      className="workspace-select-option"
      role="option"
      aria-label={option.label}
      aria-selected={option.value === value}
      data-active={index === activeIndex}
      title={option.label}
      onPointerMove={() => setActiveIndex(index)}
      onClick={() => selectOption(index)}
    >
      <span className="workspace-select-option-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24">{className === "workspace-file-select"
          ? <path d="M14 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9zM14 3v6h6M8 13h8M8 17h5" />
          : option.value === "json" ? <path d="M8 4H6v6l-2 2 2 2v6h2m8-16h2v6l2 2-2 2v6h-2M11 9h2m-2 6h2" />
          : <path d="M4 5h16v14H4zM7 15V9l3 3 3-3v6m3-4v4m-2-2 2 2 2-2" />}</svg>
      </span>
      <span className="workspace-select-option-text">
        <span>{option.label}</span>
        {option.description ? <small>{option.description}</small> : null}
      </span>
      <svg className="workspace-select-check" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6" /></svg>
    </div>)}</div>, document.body) : null}
  </>;
}
