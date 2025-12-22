# Read the Docs Setup Guide

This document provides instructions for registering pyKOSMOS++ on Read the Docs.

## Current Status

✅ `.readthedocs.yaml` configuration file is present and configured
✅ Documentation builds successfully with Sphinx
✅ GitHub repository is public and accessible

## Registration Steps

### 1. Sign in to Read the Docs

Visit https://readthedocs.org/ and sign in with your GitHub account.

### 2. Import the Project

1. Click "Import a Project" from your dashboard
2. Select the `gkhullar/pykosmospp` repository
3. Click "+" to import

### 3. Configure Build Settings

The project should use the settings from `.readthedocs.yaml`:

- **Name**: pyKOSMOS++ (or pykosmos-spec-ai)
- **Repository URL**: https://github.com/gkhullar/pykosmospp
- **Default Branch**: main
- **Language**: Python
- **Programming Language**: Python 3.10

### 4. Enable GitHub Webhook

Read the Docs automatically creates a webhook, but verify:

1. Go to your GitHub repository settings
2. Click "Webhooks" in the left sidebar
3. Verify webhook URL: `https://readthedocs.org/api/v2/webhook/...`
4. Ensure webhook is active and receiving events

### 5. Configure Advanced Settings

In the Read the Docs project settings:

**Privacy Level**: Public (recommended for open-source projects)

**Build Options**:
- ✅ Build pull requests
- ✅ Build on every commit
- ✅ Build documentation for tags

**Notifications**:
- Configure email notifications for build failures

### 6. Custom Domain (Optional)

If you want a custom domain like `docs.pykosmospp.org`:

1. Add domain in Read the Docs settings
2. Update DNS CNAME record to point to `readthedocs.io`

## Expected Build Behavior

- **Main branch commits**: Automatically build and deploy to `latest`
- **Tagged releases**: Build version-specific documentation (e.g., `v0.1.0`)
- **Pull requests**: Build preview documentation for review

## Troubleshooting

### Build Fails on Read the Docs

Check the build logs for errors. Common issues:

1. **Missing dependencies**: Ensure `docs/requirements.txt` is complete
2. **Import errors**: Verify package imports work in clean environment
3. **Sphinx warnings treated as errors**: Set `fail_on_warning: false` in `.readthedocs.yaml`

### Webhook Not Triggering

1. Check webhook delivery history in GitHub settings
2. Verify webhook secret matches Read the Docs configuration
3. Re-sync repository in Read the Docs project settings

## Documentation Links

After successful setup, documentation will be available at:

- **Latest**: https://pykosmos-spec-ai.readthedocs.io/en/latest/
- **Stable**: https://pykosmos-spec-ai.readthedocs.io/en/stable/
- **Version-specific**: https://pykosmos-spec-ai.readthedocs.io/en/v0.1.0/

## Badge for README

Add this badge to README.md to show build status:

```markdown
[![Documentation Status](https://readthedocs.org/projects/pykosmos-spec-ai/badge/?version=latest)](https://pykosmos-spec-ai.readthedocs.io/en/latest/?badge=latest)
```

## Resources

- **Read the Docs Documentation**: https://docs.readthedocs.io/
- **Configuration File Reference**: https://docs.readthedocs.io/en/stable/config-file/v2.html
- **Build Troubleshooting**: https://docs.readthedocs.io/en/stable/builds.html

---

**Note**: This setup needs to be completed by the repository owner (gkhullar) as it requires admin access to the GitHub repository and Read the Docs account.
