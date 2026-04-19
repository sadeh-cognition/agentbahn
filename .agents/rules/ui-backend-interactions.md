---
trigger: always_on
---

All UI/TUI interactions with data should be done via the backend HTTP API.
In the TUI, all business logic should be extracted into functions which can be used without the TUI. This will improve testability and separation of concerns.
In the TUI, When calling the backend API reuse the ninja schemas that define the endpoint's request payload type. Also, reuse the response schema to parse the response from the HTTP API. The goal is to ensure the shape of data sent to the backend and returned from the backend is consistent across the application and to avoid writing duplicate code for parsing and validating data.

