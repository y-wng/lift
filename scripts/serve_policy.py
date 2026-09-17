"""Serve a trained LIFT, original pi05, or residual checkpoint over WebSocket.

Usage (replace CONFIG and CHECKPOINT_STEP with the training config and output):
    uv run python scripts/serve_policy.py --config-name CONFIG --checkpoint-dir CHECKPOINT_STEP

--checkpoint-dir is the step directory containing both params/ and assets/,
not the params directory accepted by training initialization flags. Reuse the
training model config, including use_force_history=False for its ablation.
Residual-only weights additionally require the config's weight_loader to point
to the same frozen base used during training.

The default bind address is 127.0.0.1:8000. --host and --port can change it;
only expose the unauthenticated service on a trusted network. --default-prompt
supplies an instruction if the client omits one. --record saves policy I/O to
policy_records/ in the working directory.

Use packages/openpi-client/src/openpi_client/websocket_client_policy.py to send
observations. The single-arm keys and output action layout are defined in
src/openpi/policies/single_iphone_flexiv_policy.py. Robot control, sensing, and
execution of the returned action chunks are the client's responsibility.
"""

import dataclasses
import logging

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: str
    default_prompt: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    record: bool = False


def create_policy(args: Args) -> _policy.Policy:
    return _policy_config.create_trained_policy(
        _config.get_config(args.config_name), args.checkpoint_dir, default_prompt=args.default_prompt
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    logging.info("Serving %s on %s:%s", args.config_name, args.host, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port, metadata=policy_metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
