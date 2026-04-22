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

        payload = {
            "job_id": context.flow_id,
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
            callback_info={"job_id": context.flow_id, "webhook": webhook},
        )
