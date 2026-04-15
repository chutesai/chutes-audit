## Chutes auditor
This system is designed to help prove fairness of distribution of requests from the validator to the miners through a variety of mechanisms. There will be additional features, but the current implementation goes a long way in accomplishing this goal. Validators can use it to independently verify audit data and set weights; others can run it to audit subnet activity.

In order to fully verify all blocks in the 7 day audit window, you must either use an archive subtensor node, or a local subtensor with `--state-pruning 60000` to ensure you keep sufficient blocks to verify against the `set_commitment` calls.

### Recommended machine specs
- 16+ CPU cores
  - Note that single core performance is very critical here as well, please select a fast CPU (3+ghz for example)
  - If you must make a trade-off between more cores vs fewer but faster cores, you will likely benefit from choosing faster (but adjust the docker-compose yaml accordingly on worker counts/etc. to match your specs)
- 64gb+ RAM ideally, but can function with lesser amount
  - Adjust the docker compose yaml accordingly, e.g. shm size, postgres cache sizes, etc.
- 1TB+ (fast) disk
  - This should really but a locally attached NVMe/SSD, preferably a raid 1 or 10 for some resiliency
  - Try to avoid network attached storage or shared drives which may encounter high latency, low IOPS, etc.
  - Disk performance is quite important here since the majority of calculations take place in postgres.

You can use lower spec machines but performance will be sub-optimal of course, and see below - likely need to tweak settings.

#### Postgres configuration
Based on the number of cores, RAM, etc., you will likely wish to change the postgres command in the docker-compose.yml file increasing or decreasing various values accordingly.

### Initial sync
Before you begin, be sure to copy config/config.yml.example to config/config.yml and make any adjustments/changes you would like, particularly if you are a validator and wish to set weights.

The first time you run this, it takes an extremely long time to sync, upwards of 8+ hours. Using faster disks/CPU and more RAM will help. Until the initial sync is finished, you will not set weights.

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
Any time new audit data is available, the auditor downloads the committed JSON reports (validator instance audit data and compute history) and can calculate incentives from that data. It can compare the calculated distribution to the current metagraph weights. There will always be some minor discrepancies due to weight copiers and a few seconds to minutes of potential gap between chain state and the latest audit exports, but it should be extremely close.
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
  enabled: true
  ss58_address: 5GerCEPSx22bmr5Wnm2wj87SSpZiVvhVqFUrGG5795XkUbjr
  secret_seed: '0x971c2a6674d0861ade72297d11110ce21c93734210527c8f4c9190c00139ce20'
```
### Running the auditor
Before attempting to run the auditor, be sure to go through the `config/config.yml` file and make any changes you wish. The main change for validators is to configure the `set_weights` section with your SS58 and hotkey seed if you want the auditor to set weights on your behalf.

Once you have the config file updated, there are two ways to run it:


__Option 1:__ Run the autoupdater (Ensure you have python installed)

**Install pm2 if needed**
```bash
apt-get install -y -qq nodejs npm
npm i -g -q pm2
apt-get install -y -qq jq
```

**Run the autoupdater**
```bash
pm2 delete autoupdates || true && pm2 start --name "autoupdates" "python utils/autoupdater.py"
```


__Option 2:__ just use docker compose
```bash
docker compose up --build auditor
```


__Option 3:__ install python, poetry, etc., and use it without docker
You will need poetry for dependency management (or you can parse out requirements from `pyproject.toml`), e.g. `curl -sSL https://install.python-poetry.org | python3 -`
Make sure you have postgres running locally (which you can do using the provided docker compose file if you wish), and set the `POSTGRESQL` environment variable, e.g.: `export POSTGRESQL='postgresql+asyncpg://user:password@127.0.0.1:5432/chutes_audit'`






