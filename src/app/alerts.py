"""Alert generation: streets crossing the alert threshold at a given hour."""
import json


def generate_alerts(scored: list[dict], level: str = "alert") -> list[dict]:
    return [dict(street_id=s["id"], lat=s["mid"][0], lon=s["mid"][1],
                 risk=s["risk"], band=s["band"],
                 message=f"Flood alert on street {s['id']}: risk {s['risk']}/100. "
                         f"Avoid low-lying stretches and move vehicles to higher ground.")
            for s in scored if s["band"] == level]


def alerts_for_hour(timeline: dict, hour: int, level: str = "alert") -> list[dict]:
    return generate_alerts(timeline.get(hour, []), level)


def render_sms(alerts: list[dict], limit: int = 5) -> str:
    """Mock SMS/push payloads a dispatcher (Twilio/Gupshup etc.) would send."""
    lines = [f"[mock SMS] {a['street_id']}: {a['message']}" for a in alerts[:limit]]
    if len(alerts) > limit:
        lines.append(f"... and {len(alerts) - limit} more")
    return "\n".join(lines) if lines else "No alerts at this hour."


if __name__ == "__main__":
    with open("data/processed/timeline.json") as f:
        tl = json.load(f)
    peak = max(tl, key=lambda h: sum(1 for _ in tl[h] if True))
    h = max(tl, key=lambda h: sum(1 for s in tl[h] if s["band"] == "alert"))
    print(f"Peak alert hour: {h}")
    print(render_sms(alerts_for_hour(tl, int(h))))
