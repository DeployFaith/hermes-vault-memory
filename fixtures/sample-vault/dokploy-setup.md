# Dokploy deployment notes

Mount the Markdown vault at `/vault` and keep `/data` on persistent storage.
The service indexes Markdown files and searches terms like vault mount, compose, and deploy.

When the vault mount is missing the service fails fast on startup, which helps catch bad deployment settings.
