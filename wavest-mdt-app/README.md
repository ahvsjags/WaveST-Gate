# WaveST-MDT

Bilingual, privacy-safe spatial pathology workstation prototype for multidisciplinary oncology review.

## Public workstation

The de-identified research demo is hosted at:

<https://ahvsjags.github.io/WaveST-Gate/>

Run `START_PUBLIC.cmd` to open the hosted workstation. It remains available when the local computer is off.

## Local run

```powershell
pnpm install
pnpm prepare:data
pnpm dev
```

Open `http://localhost:5173`. The development server binds to `0.0.0.0`, so devices on the same network can use the computer's LAN address while the server is running.

For normal use after installation, run `START_LOCAL.cmd`. Devices on the same network can use `http://<computer-lan-ip>:4173` while the local server is running. `STOP_WAVEST_MDT.cmd` stops a server launched by the local start script.

## Production preview

```powershell
pnpm build
pnpm preview
```

The prototype bundles only the de-identified public project outputs prepared by `scripts/build-demo-data.mjs`. It does not copy or expose the private hospital intake folder.

The GitHub Pages deployment is published from the isolated `gh-pages` branch. Run `pnpm build` before publishing a new verified release.
