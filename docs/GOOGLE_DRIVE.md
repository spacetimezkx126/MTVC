# Upload checkpoints to Google Drive

## Browser method

Download the checkpoint archive to a local computer. Open Google Drive in a browser, choose New > File upload, and select the archive and its checksum file. Upload the code archive separately if desired. Keep checkpoints outside the GitHub repository.

## Server method with rclone

Install rclone using its official installation instructions. On the server run:

```bash
rclone config
```

Create a remote named `gdrive`, choose the Google Drive backend, and follow the authorization prompts. On a headless server answer No when asked whether to use a browser. Run the exact `rclone authorize` command printed by the server on a computer with a browser, then paste the returned token into the server prompt. Do not put the token or rclone.conf in the GitHub repository.

From the code repository root:

```bash
bash scripts/upload_checkpoints.sh ../MTVC_checkpoints gdrive:MTVC/verified-checkpoints
```

The script uses `rclone copy` and then `rclone check --one-way`. To upload a prepared archive instead:

```bash
rclone copy ../MTVC_release_artifacts/MTVC_checkpoints.tar.gz gdrive:MTVC/ --progress
rclone copy ../MTVC_release_artifacts/SHA256SUMS gdrive:MTVC/ --progress
```

Official references: https://rclone.org/drive/ and https://rclone.org/remote_setup/ . Authentication requires your Google account; the release scripts do not log in or upload until you explicitly run them.
