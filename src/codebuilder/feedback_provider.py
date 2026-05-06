import logging
import os

import requests
from crewai.flow import (
    ConsoleProvider,
    HumanFeedbackPending,
    HumanFeedbackProvider,
    PendingFeedbackContext,
)

log = logging.getLogger(__name__)


class WebhookFeedbackProvider(HumanFeedbackProvider):
    """Async feedback provider: notify Flask by HTTP, then pause the flow.

    The webhook URL is read from the `CODEBUILDER_APPROVAL_WEBHOOK` env var.
    If unset, the provider falls back to the blocking ConsoleProvider so
    running the flow in a local terminal still works.
    """

    def __init__(self, webhook_env: str = "CODEBUILDER_APPROVAL_WEBHOOK"):
        self.webhook_env = webhook_env
        self._fallback = ConsoleProvider()

    def request_feedback(self, context: PendingFeedbackContext, flow) -> str:
        webhook = os.environ.get(self.webhook_env)
        if not webhook:
            log.warning(
                "%s not set — falling back to console feedback. "
                "Set it to enable async HITL via Flask.",
                self.webhook_env,
            )
            return self._fallback.request_feedback(context, flow)

        # Caller-supplied session_id (decoupled from flow_id so AMP OTel
        # traces correlate — see CON-101). May be empty when running locally
        # via the console fallback; the frontend correlates by session_id
        # when present and falls back to flow_id otherwise.
        session_id = getattr(getattr(flow, "state", None), "session_id", "") or ""

        payload = {
            "session_id": session_id,
            "flow_id": context.flow_id,
            "job_id": context.flow_id,  # backward-compat alias for flow_id
            "flow_class": context.flow_class,
            "method_name": context.method_name,
            "message": context.message,
            "method_output": PendingFeedbackContext._make_json_safe(context.method_output),
            "outcomes": list(context.emit or []),
            "default_outcome": context.default_outcome,
        }
        try:
            requests.post(webhook, json=payload, timeout=5)
        except requests.RequestException as exc:
            log.error("Feedback webhook POST failed: %s", exc)

        raise HumanFeedbackPending(
            context=context,
            callback_info={
                "session_id": session_id,
                "flow_id": context.flow_id,
                "job_id": context.flow_id,
                "webhook": webhook,
            },
        )
