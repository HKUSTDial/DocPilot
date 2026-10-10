from pipelines.training.convert_docpilot_logs_to_agentic_training_official import render_action_route, render_focus_tool


def step(action_type, args=None, payload=None):
    return {
        "action": {"action_type": action_type, "args": args or {}},
        "observation": {"action_type": action_type, "payload": payload or {}},
    }


def main():
    assert render_action_route(
        step("navigate", {"pages": "1-19"}, {"page_set": list(range(1, 11))})
    ).endswith("target_pages: \\boxed{[1-10]}")
    assert render_action_route(
        step("extract", {"candidate_pages": [{"page": 14}, {"page": 16}]})
    ).endswith("target_pages: \\boxed{[14, 16]}")
    assert render_action_route(step("focus", {"evidence_ids": ["page14::screenshot"]})).endswith(
        "evidence_ids: \\boxed{[page14::screenshot]}"
    )
    assert render_action_route(step("solver")) == "next_action: solve"
    assert render_action_route(step("answer")) == "next_action: answer"
    assert render_focus_tool(
        {},
        {
            "payload": {
                "operations": [
                    {
                        "tool": "zoom",
                        "evidence_id": "1032373.pdf::page::1::page_figure::2",
                        "region": {"bbox": [0.0, 0.0, 1.0, 0.5]},
                    }
                ]
            }
        },
        {},
    ).endswith(
        'operation: \\boxed{{"tool":"zoom","evidence_id":"1032373.pdf::page::1::page_figure::2","bbox":[0.0,0.0,1.0,0.5]}}'
    )


if __name__ == "__main__":
    main()
