import { useEffect, useState } from "react";

import { api } from "../api/client";
import type { StatusEvent } from "../types";

/**
 * Open an EventSource to the experiment's SSE stream and surface the latest
 * status event. Cleans up on unmount or when `id` changes; closes on terminal
 * status so the browser does not retry indefinitely.
 */
export function useExperimentEvents(id: string | null): StatusEvent | null {
  const [event, setEvent] = useState<StatusEvent | null>(null);

  useEffect(() => {
    if (!id) {
      setEvent(null);
      return;
    }
    setEvent(null);

    const url = api.experiments.eventsUrl(id);
    const source = new EventSource(url);

    source.addEventListener("status", (ev) => {
      try {
        const payload = JSON.parse((ev as MessageEvent).data) as StatusEvent;
        setEvent(payload);
        if (payload.status === "completed" || payload.status === "failed") {
          source.close();
        }
      } catch {
        // Ignore parse errors; the server controls payload format.
      }
    });

    source.onerror = () => {
      source.close();
    };

    return () => {
      source.close();
    };
  }, [id]);

  return event;
}
