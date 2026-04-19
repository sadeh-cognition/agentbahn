---
trigger: always_on
---

To create CLI commands use `django-click`. `django-click` is documented here: <https://github.com/django-commons/django-click>
For the TUI user interface use the python `rich` package.
For the front-end use HTMX and Tailwind CSS.
The backend is a Django HTTP API implemented using `django-ninja`.
When interacting with the backend, always use the HTTP API.
Always use the HTTP API for fetching, updating, or deleting data.
When writing Django code, try your hardest not to use Django signals.
