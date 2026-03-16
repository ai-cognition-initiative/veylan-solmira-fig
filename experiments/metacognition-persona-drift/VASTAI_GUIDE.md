# Vast.ai Guide for Claude

## Lesson 1: Command syntax requires "instance" keyword

```bash
# WRONG
vastai reboot <id>

# CORRECT
vastai reboot instance <id>
```

## Lesson 2: Rebooting does NOT fix SSH key issues

If SSH fails with "Permission denied (publickey)" after the key is attached, rebooting the instance does NOT help. The instance comes back up but SSH still fails.

## Lesson 3: Check which SSH keys are registered AND attached

```bash
# Keys registered with account
vastai show ssh-keys

# Keys attached to specific instance
vastai show instance <id> --raw | python3 -c "import json,sys; d=json.load(sys.stdin); print('SSH keys attached:', d.get('ssh_key_ids', 'none'))"
```

If `ssh_key_ids` is empty/none, SSH will ALWAYS fail even if you ran `vastai attach ssh`.

## Lesson 4: Attach SSH key syntax IS correct

```bash
vastai attach ssh <instance_id> "$(cat path/to/key.pub)"
```

This returns `{'success': True, 'msg': 'SSH key added to instance.'}` but SSH can STILL fail with "Permission denied". The attach succeeding does not guarantee SSH will work.

## Lesson 5: Two .env files cause confusion

There were two .env files:
- `../.env` (parent directory, outside repo)
- `.env` (repo root, deleted)

`find_dotenv(usecwd=True)` loads the nearest one. If running from the experiment dir, it would load the repo .env FIRST, ignoring the parent.

**Solution**: Delete repo .env. Keep only the parent .env.

## Lesson 6: VAST_SSH_KEY must be a file path (not key content)

SSH commands (`ssh -i`, `scp -i`) need a KEY FILE, not the key content directly.

```
# WRONG - SSH can't use this
VAST_SSH_KEY=ssh-ed25519 AAAAC3NzaC1lZDI1NTE5...

# CORRECT - path to private key file
VAST_SSH_KEY=~/.ssh/vast-key
```

## Lesson 7: Use the correct API key

The Rethink Priority API key is stored at:
your vast.ai API key file (set `VASTAI_API_KEY` in environment)
