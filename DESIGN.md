# Design System: HIPAA Predictor — Command Center
**Project ID:** 18287675361302976401

## 1. Visual Theme & Atmosphere

The design embodies a **Clinical-Precision Meets Modern Tech** aesthetic — clean, airy, and confidence-inspiring. Light backgrounds create breathing room, while a deep teal sidebar anchors the interface with authority. The overall density is **medium-to-low**: generous whitespace between cards, clear visual hierarchy, and zero clutter. The mood is calm professionalism — a surgeon's dashboard, not a social media feed.

The landing page adds dramatic flair with bold, uppercase typography, a dark "terminal-style" preview window, and punchy call-to-action sections. Together the dashboard and landing page communicate: **enterprise-grade reliability wrapped in modern web aesthetics**.

## 2. Color Palette & Roles

| Descriptive Name | Hex Code | Functional Role |
|---|---|---|
| **Deep Command Teal** | `#0D2B3E` | Sidebar background, primary brand anchor |
| **Vivid Teal-Cyan** | `#00BCD4` | Primary accent: active nav items, CTA buttons, status indicators, links |
| **Whisper Cloud White** | `#F5F7FA` | Main content area background — a barely-there warm grey |
| **Pure White** | `#FFFFFF` | Card surfaces, elevated containers |
| **Ink Charcoal** | `#1A1A2E` | Primary heading text, hero bold typography |
| **Slate Grey** | `#6B7B8D` | Secondary/muted text, labels, subtitles |
| **Alert Coral-Red** | `#E74C3C` | Critical alerts, high-risk badges, "Requires attention" indicators |
| **Warm Amber** | `#F59E0B` | Elevated-risk badges, warning states |
| **Fresh Mint Green** | `#10B981` | Positive indicators: "Optimal," "+5.2%", "HIPAA Compliant" badges |
| **Soft Blush Pink** | `#FDE8E8` | Alert card pill backgrounds (light wash behind coral text) |
| **Light Fog Grey** | `#E5E7EB` | Borders, dividers, table row separators |

## 3. Typography Rules

- **Font Family:** Space Grotesk (geometric sans-serif) — used globally for both headlines and body text
- **Display/Hero Text:** Bold (700–800 weight), uppercase, tight letter-spacing (`-0.02em`), creating a bold, tech-forward impact. Sizes range from `3rem` to `4rem` for landing page heroes
- **Dashboard Headings:** Semi-bold (600), sentence case, `1.25rem–1.5rem`. Clean and scannable
- **Stat Numbers:** Bold (700), large (`2.5rem–3rem`), creating immediate visual dominance in metric cards
- **Body Text:** Regular weight (400), `0.875rem–1rem`, generous line height (`1.6`) for readability
- **Label/Caption Text:** Medium (500), uppercase, `0.7rem`, letter-spacing `0.08em` — used for category labels like "TOTAL PREDICTIONS", "SYSTEM STATUS"
- **Monospace Accents:** For telemetry data and timestamps, a subtle shift to monospace reinforces technical precision

## 4. Component Stylings

* **Buttons:**
  - **Primary CTA:** Pill-shaped (fully rounded, `border-radius: 999px`), Vivid Teal-Cyan background, white text. Medium padding (`0.75rem 1.5rem`). Subtle scale-up on hover (`transform: scale(1.02)`)
  - **Secondary/Outline:** Pill-shaped, transparent background, thin border in Slate Grey, dark text. On hover, fills with a whisper of teal
  - **Sidebar "New Analysis":** Full-width pill button with `+` icon, teal background, positioned at sidebar bottom

* **Cards/Containers:**
  - Gently rounded corners (`border-radius: 12px`)
  - Pure White background on Whisper Cloud base
  - Whisper-soft box-shadow (`0 1px 3px rgba(0,0,0,0.08)`) — barely-there elevation
  - **Stat Cards:** Row of 4, each with an uppercase teal/coral label at top, a massive bold number, a small status pill below, and a watermark icon in the top-right corner (low-opacity teal)

* **Inputs/Forms:**
  - Rounded rectangle (`border-radius: 8px`)
  - Light grey border (`1px solid #E5E7EB`), white background
  - On focus: teal border glow (`box-shadow: 0 0 0 3px rgba(0,188,212,0.15)`)

* **Sidebar Navigation:**
  - Full-height, Deep Command Teal background
  - Nav items: white text, `0.9rem`, icon + label. Vertical stack with `0.5rem` gaps
  - **Active item:** Vivid Teal-Cyan background pill, bold white text, left-aligned icon
  - Section dividers: uppercase grey labels ("SYSTEMS")
  - Badge: small coral-red circle with count (e.g., "12") on Patient Alerts

* **Alert/Priority Cards:**
  - Avatar circle with initials (Deep Command Teal background, white text)
  - Risk badge: small uppercase pill — coral red for "HIGH RISK", amber for "ELEVATED"
  - Detail pill: soft blush pink background with coral text description
  - Divider lines between alert items

* **Charts:**
  - Clean line chart with teal stroke, subtle gradient fill below the line
  - Minimal axes: light grey grid lines, slate text for labels
  - Time-range toggle: pill buttons (7 DAYS / 30 DAYS / 90 DAYS) — active one fills teal

* **Progress Bars:**
  - Rounded track (`border-radius: 4px`), light grey background
  - Filled segment in Vivid Teal-Cyan
  - Label pair: category on left, percentage on right

* **Tables:**
  - Clean, minimal borders — only bottom border on rows (`1px solid #F0F0F0`)
  - Header row: uppercase, small, Slate Grey text
  - Hover state: subtle row highlight (`background: #F9FAFB`)

## 5. Layout Principles

- **Sidebar + Main Content:** Fixed-width sidebar (`240px`) on the left, flex-grow main content area. Sidebar is full-viewport height with sticky positioning
- **Top Bar:** Horizontal strip above main content: status indicator (green dot + "OPERATIONAL"), search input (centered), theme/notification icons, and user profile (avatar + name) right-aligned
- **Card Grid:** Stat cards in a 4-column grid with `1rem` gaps. Below, a 2/3 + 1/3 split for chart area and alerts panel
- **Landing Page Sections:** Full-width sections with comfortable vertical padding (`5rem–6rem`). Content max-width `1120px`, centered. Hero uses a 50/50 text-image split
- **Generous Whitespace:** Minimum `1.5rem` padding inside cards, `1.5rem` gap between card rows. The design breathes — nothing feels cramped
- **Responsive Strategy:** At narrow viewports, sidebar collapses to an icon-only rail or hamburger menu. Stat cards stack to 2-column then 1-column. Charts and alerts go full-width stacked
