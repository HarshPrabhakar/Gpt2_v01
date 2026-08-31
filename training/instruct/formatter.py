from __future__ import annotations


USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant: "
SYSTEM_PREFIX = "System: "


def normalize_messages(messages):
    """
    Validate and normalize a list of chat messages.

    Accepted roles:
        system
        user
        assistant
    """

    normalized = []

    if not isinstance(messages, list):
        return normalized

    for message in messages:

        if not isinstance(message, dict):
            continue

        role = str(
            message.get("role", "")
        ).strip().lower()

        content = str(
            message.get("content", "")
        ).strip()

        if role not in {
            "system",
            "user",
            "assistant",
        }:
            continue

        if not content:
            continue

        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized


def role_prefix(role: str) -> str:

    role = role.strip().lower()

    if role == "user":
        return USER_PREFIX

    if role == "assistant":
        return ASSISTANT_PREFIX

    if role == "system":
        return SYSTEM_PREFIX

    raise ValueError(
        f"Unsupported role: {role}"
    )


def format_conversation(messages) -> str:

    parts = []

    for message in normalize_messages(
        messages
    ):

        parts.append(
            role_prefix(
                message["role"]
            )
            + message["content"]
        )

    return "\n\n".join(parts)