# Web chat

## Summary

The web chat lets you talk to one running companion in a browser.
It looks like a normal messenger: you log in with a name, you see the full history, and you type at the bottom.
The server runs in the app process on `127.0.0.1` and reads the live memory store.
Every message on the page is an accepted record, so a restart shows the same history.

Parent: [operator service](console.md).
Related: [public dialogue](public-dialogue.md), [publication and recovery](public-publication.md), and [images in records](multimodal-records.md).
Code: [`pm.app.web_chat`][], [`pm.app.cli`][], and [`workflows.public_speech.user_input.UserInputProxy`][].

## Start

You need a character home folder with a `ghost.json` checkpoint.
`character init` creates this folder.

1. Start the app with the home folder and your model settings:

    ```powershell
    uv run --locked python main.py --home <home> --config config.alpha.openrouter.yaml
    ```

2. Open the URL that the app prints, for example `Web chat: http://127.0.0.1:8770/`.
3. Type your name and select **Start chatting**.
4. Type a message. Push Enter to send it. Push Shift+Enter for a new line.
   To send images, select **+**, paste an image, or drop image files on the chat.
   Each image shows a preview chip. Select its **×** to remove it.
5. To stop the app, push Ctrl+C in the terminal.

`--web-port <port>` changes the port. The default is `WEB_CHAT_PORT` in `pm/constants.py`.
`--cognition dummy` runs the labeled fixture without model calls.
The terminal also accepts input as the user. Replies also go to the terminal.

The home folder supplies the persona and the runtime root:

| Path | Use |
| --- | --- |
| `<home>/character.yaml` and `<home>/personality.json` | Persona. The app reads them at each start. |
| `<home>/ghost.json` | Accepted memory. The app loads it and saves every commit to it. It must exist. |
| `<home>/runtime/` | Runtime root: kernel folders, logs, and model call logs. |

`--home` excludes `--seed`, `--checkpoint`, `--ephemeral`, and `--runtime-root`.
`--config` stays the model settings.
Without `--home`, the terminal session works as before.
`--web` starts the web chat in that terminal session too.

## Login and speakers

The page keeps the login name in the browser's `localStorage`.
**Change name** at the top selects a different name.
The name is the speaker of each message that you send.

- The configured user name is the user. These messages are the same as terminal input.
- Any other name is a different person.
  The first message of a new name creates a `Human` entity with that name in the same commit.
  Later messages of that name use the same entity.
  Main cognition puts them into the stimulus group of that entity, so the person gets their own workspace shard.
  The reply addresses that entity.
- The companion's name, or an alias of its entity, is rejected.

Each human message has `SourceKind.User` provenance, so publication resolves its exchange like a user exchange.
The page shows your own messages on the right and all other lines on the left.
Lines on the left show the speaker name.

## API

All bodies are JSON. The models are in [`pm.app.web_chat`][].

| Request | Result |
| --- | --- |
| `GET /api/session` | [`ChatSession`][pm.app.web_chat.ChatSession]: companion name, user name, poll interval |
| `GET /api/messages?after=<id>` | [`MessagePage`][pm.app.web_chat.MessagePage]: messages with an ID above `after`, `last_id`, and `thinking` |
| `POST /api/messages` with `{"name": ..., "text": ..., "images": [...]}` | [`PostResult`][pm.app.web_chat.PostResult]: the exchange correlation ID |
| `GET /api/media/<record_id>/<image_id>` | The bytes of one image of one public dialogue record, with its MIME type |

Each [`ChatMessage`][pm.app.web_chat.ChatMessage] has `id`, `speaker`, `text`, `timestamp` (UTC), `from_companion`, `exchange_status`, and `images`.
`images` lists the image ids of the record. `text` is the content without the image tags. The history never contains base64.
The page shows each image as a thumbnail from `/api/media/...`. A click opens the full image.

Each entry of `images` in a POST is a [`PostImage`][pm.app.web_chat.PostImage]: `media_type`, `data_b64`, and an optional file `name`.
A message needs text, images, or both. It holds at most `WEB_CHAT_MAX_IMAGES` images of at most `WEB_CHAT_MAX_IMAGE_BYTES` each.
The server checks the magic bytes against `media_type`. Invalid data, a wrong type, or an image tag in the text gives status 400.
The body limit is `WEB_CHAT_MAX_IMAGE_BODY_BYTES`, because base64 makes the data one third larger.
The history contains every public Dialogue feature, imported history included.
The order is world time, then ID.
A new message always has a higher ID and a later time, so the page appends the lines after its `last_id`.

`thinking` is true while a human message after the newest companion line has `exchange_status` `pending`.
The page then shows "<companion> is thinking…".

A rejected request returns an `error` text with status 400, 403, 404, or 413.
An exception in a request is a runtime bug: the server passes it to the application error handler, answers 500, and the app stops.

## Reads and writes

[`read_messages`][pm.app.web_chat.read_messages] holds the store lock while it copies the dialogue lines.
A commit holds the same lock while it validates, writes `ghost.json`, and installs the new records.
So a read sees the state before or after a commit, never a part of one.

A POST calls `UserInputProxy.submit(text, speaker=name, images=[...])`.
Submit attaches the images to the Dialogue feature and appends their tags (see [images in records](multimodal-records.md)).
Submit commits the message before it puts the `user.input` event on the bus, so the next poll shows the message.
The page polls every `WEB_CHAT_POLL_MS` milliseconds and once directly after a send.

## Persistence and restart

Every accepted commit writes `ghost.json` atomically: a temporary file, then a replacement.
This includes each user message, each cycle, each reply, and each delivery mark.
A crash or a kill keeps the last complete checkpoint.

Ctrl+C, SIGTERM, and Ctrl+Break stop the app in this order:

1. The web server stops and waits for running requests.
2. The modules stop and join.
3. The managed runs stop.
4. A clean shutdown event commits to `ghost.json`.

Closing the browser tab changes nothing on the server.
A cycle that did not commit before the stop is not accepted state, and the restart does not show it.

At the next start, the app loads `ghost.json`.
The tick counter, the cycle states, the GWT turns, and the dialogue continue from the saved values.
The next main cycle receives the last saved cycle state as `previous_state`.
The startup event records the offline gap.

## Limits

- The server has no authentication. Anyone who can open the loopback URL can speak as any name.
- The login name is not an identity check. Two browsers with the same name are the same speaker.
- `thinking` is an estimate. A spontaneous companion line ends it before the reply comes. A message that cognition chooses not to answer keeps it until the next companion line.
- The dummy fixture does not resolve exchanges. Its messages stay `pending`, but its reply still ends `thinking`.
- Each poll scans all features under the store lock. A very large store makes each poll slower.
- A closed console window gives Windows about 5 seconds for the stop. A long model call can cut it; the last commit still survives.
- The page shows only public dialogue. It does not show thoughts, tasks, or workflow events.
- Images in companion replies are not supported. Only human messages carry images.
