## Chutes auditor
This system is designed to help prove fairness of distribution of requests from the validator to the miners through a variety of mechanisms. There will be additional features, but the current implementation goes a long way in accomplishing this goal, and can be run by *anyone* interested, not just validators with a ton of stake.

### Synthetics to help guarantee completeness of audit exports/weights
If configured with synthetics.enabled and a valid API key, the auditing system will continuously generates synthetics for all of the various standard templates. For each request, it will use the chutes invocation tracing system to get exactly which miner the request was routed to, any errors that trigger retry to another miner, etc., and verify that each and every one of those synthetics show up in the audit exports from the validator. Invocations should ALWAYS show up in the invocation exports the audit system pulls.
Example trace:
```
Attempting to perform synthetic task: task_type='image'
Invoking chute.name='AnimePastelDream' at https://chutes-animepasteldream.chutes.ai/generate
2025-02-10T15:51:19.216856 [invocation_id=f225adc1-da28-4de9-8640-a2993cc8c1f7 child_id=3e1d4eec-0ac7-4ae5-a5a0-04e54108cfe1 chute_id=791ea6b9-04b6-5f75-9f42-b30259553277 function=generate]: attempting to query target=d669c3e4-1e1f-41b5-b2ab-0f5dc330ca58 uid=49 hotkey=5EHhZEcksG5mds1j5HvKLhk4DZSe5AUkqFeeiDD23VYS8WG1 coldkey=5FhEkzwyGXdoktoE2i3VJWH12mW29YsQ2hBWC5gJMCWCxVyk
[image]
```
The outputs are all rendered by default (including playing audio and rendering images!), but you can toggle rendering off for each synthetic type.
See `config/config.yml` for additional synthetics configuration options.
### Comparing validator metrics to miner self-reported metrics
One of the functions of the auditing script is to actually compare the miner reported metrics from their prometheus against the invocation counts from the validator.  Prometheus metrics will always be a bit wrong in comparison to something like the validator postgres database, simply because the way the prometheus server is configured in the extras playbook, it's actually just a stateless pod with no stateful storage (so if it restarts the data starts over). Also potentially issues from wireguard/calico/etc in actually scraping the metrics. Even so, even with those issues, you can see the audit system shows agreement in summary metrics reported by miners with the invocation exports at ~90% (and the disagreement was underreporting from the prometheus stats as somewhat expected).

```
Miner 5FhMaRd59y5nyDEtCz1JMMEMZzAGimtmC8m5AfCeXVE3vzCx has full audit report coverage [601200 seconds]
Miner 5Fpw5S6drw26vujZVoCwjQWfmgv6Vp82Jqj2Fxw3dGaZsrGt has full audit report coverage [601200 seconds]
Miner 5FvLzisiVtovB8zAuX3Jqne9T3gqiEyCbRx4Z4Ly6ETc3WXg has full audit report coverage [601200 seconds]
...
Miner 5FhMaRd59y5nyDEtCz1JMMEMZzAGimtmC8m5AfCeXVE3vzCx reported 103655 vs audit 108817: agreement ratio 0.9526
Miner 5Fpw5S6drw26vujZVoCwjQWfmgv6Vp82Jqj2Fxw3dGaZsrGt reported 63950 vs audit 65077: agreement ratio 0.9827
Miner 5FvLzisiVtovB8zAuX3Jqne9T3gqiEyCbRx4Z4Ly6ETc3WXg reported 100763 vs audit 108142: agreement ratio 0.9318
```
### Incentives calculation reproduction
Any time new audit data is available, the invocations CSV exports will be downloaded, which contain each and every invocation (along why compute multipliers, bounties, error messages, etc.) for the hour that corresponds to that audit entry.  Once the entire 7 day history of audit exports is fully loaded, the audit system can calculate and reproduce the weights locally and compare them to the current metagraph weights.  There will always be some minor discrepancies do to weight copiers and a few seconds to minutes of potential gap in what the chutes validator has vs the auditing exports, but it should be extremely close.
Example outputs:
```
Calculated incentive locally for 5F22KgAv4kvJEMcmPoWLLMkAysFUALH9JLJh9exg7QFv6s5H [  2]: 0.08938 vs actual 0.08960, delta 0.00021
Calculated incentive locally for 5EemLYa94DLwmY35g6EfLTuahnf5iHvQULjhQp3UUPyb3Tok [  3]: 0.11454 vs actual 0.11307, delta 0.00147
Calculated incentive locally for 5CaqyHE9eBPyN469MNKor8R3zoyNsQwCzMZjd51xAR66S8tF [  5]: 0.18241 vs actual 0.18234, delta 0.00007
Calculated incentive locally for 5DFurcu7b4XbArin6Rjw2Yev4AE3ScxyByGXxpatJs956eth [  6]: 0.07283 vs actual 0.07300, delta 0.00016
```
You can see here, the delta very small, < 0.2%.
### Independent weight setting as a validator
If you wish, rather than child hotkey or running a full validator with all the (expensive) bells and whistles, you can use this system to independently set weights from the audit export data. To do so, update `config/config.yml`, for example:
```yaml
set_weights:
  enabled: false
  ss58_address: 5GerCEPSx22bmr5Wnm2wj87SSpZiVvhVqFUrGG5795XkUbjr
  secret_seed: 0x971c2a6674d0861ade72297d11110ce21c93734210527c8f4c9190c00139ce20
```
### Running the auditor
Before attempting to run the auditor, be sure to go through the `config/config.yml` file and make any changes you wish. The biggest changes are:
1. Confgure api_key with a valid API key for chutes. You can register either via the CLI or through chutes.ai, then get an API key (again, either with `chutes keys create --name foo ...` or from the website).
2. If you are a validator, registered on 64, and wish to set weights, be sure to configure the `set_weights` section with your SS58 and hotkey seed.

Once you have the configuration updated, there are two ways to run it:
Option 1: install python, poetry, etc., and use it without docker
You will need to install `portaudio2` or disable audio rendering e.g. `sudo apt-get -y install libportaudio2`
You will also need poetry for dependency management (or you can parse out requirements from `pyproject.toml`), e.g. `curl -sSL https://install.python-poetry.org | python3 -`
Make sure you have postgres running locally (which you can do using the provided docker compose file if you wish), and set the `POSTGRESQL` environment variable, e.g.: `export POSTGRESQL='postgresql+asyncpg://user:password@127.0.0.1:5432/chutes_audit'`

Option 2: just use docker compose
```bash
docker compose up --build auditor
```
