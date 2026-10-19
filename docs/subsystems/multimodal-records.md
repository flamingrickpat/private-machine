# Images in records

## Summary

A record can contain images.
The record text holds a tag, `[image:<id>]`, at the position of the image.
The record metadata holds the image bytes as base64, together with an alt text.
In a prompt, the tag becomes the real image or its alt text.
The real image is the normal case. The alt text is the fallback.
Memory consolidation, reconsolidation, and embeddings always read the alt text.

Parent: [experienced memory](experienced-memory.md).
Related: [web chat](web-chat.md), [character import and reconsolidation](character-import.md), [Minecraft actor](minecraft-actor.md), and [Python chat and agent context](llm-context.md).
Code: [`pm.records.media`][], [`pm.llm.multimodal`][], and [`pm.memory.media_alt`][].

## Record shape

[`attach_images`][pm.records.media.attach_images] writes a [`RecordMedia`][pm.records.media.RecordMedia] document into `metadata["media"]`.
It returns the tags. The caller appends them to the content, one per line ([`with_tags`][pm.records.media.with_tags]).
[`record_media`][pm.records.media.record_media] is the typed read. Other code does not read the raw key.

```yaml
content: "what is this?\n[image:img-c9e20a953f976a81]"
metadata:
  media:
    schema_id: pm.record-media.v1
    images:
      - id: img-c9e20a953f976a81
        media_type: image/webp
        data_b64: "<base64 of the decoded bytes>"
        sha256: "<SHA-256 of the decoded bytes>"
        size_bytes: 4782
        width: 268
        height: 260
        alt_text: "A grey cat ..."
        alt_text_source: model        # model, user, or tool
        provenance:
          source: user_upload         # user_upload, tool, or import
          run_id: ""
          artifact_path: spoingus.webp
          server: ""                  # MCP server of a tool image
          tool: ""                    # MCP tool of a tool image
          created_at: "2026-10-19T07:19:30+00:00"
```

[`image_from_bytes`][pm.records.media.image_from_bytes] creates a [`RecordImage`][pm.records.media.RecordImage]:

- The magic bytes set the MIME type: PNG, JPEG, WebP, or GIF. A supplied type that disagrees raises `ValueError`.
- It computes the digest and the size of the decoded bytes.
- It reads the width and height from the file header, without an image library. An unusual header gives `None`.
- Data above `MEDIA_MAX_IMAGE_BYTES` and data that is not an image raise `ValueError`.

## The tag and its resolution

The image id is content addressed: `img-` and the first 16 hex digits of the SHA-256.
The same bytes therefore have the same id in every record.
A tag needs no record id, so it travels unchanged through every renderer: the story prefix, dialogue transcripts, shard tails, and summaries.

The application installs one media lookup per process ([`install_media_lookup`][pm.records.media.install_media_lookup]).
`build_app` installs a [`MediaLibrary`][pm.memory.media_alt.MediaLibrary] over its memory store and its stage model.
[`MemoryStore.find_image`][pm.memory.store.MemoryStore.find_image] returns the image from the newest accepted record that holds the id.
A tag that no record holds reads `[image: unavailable]`.
The web chat rejects user text that contains a tag, so a user cannot point at the image of another record.

## Alt text

[`complete_alt_text`][pm.memory.media_alt.complete_alt_text] asks the multimodal model for each image without alt text.
The call shows the image itself, through the normal chat client.
The prompt (`pm/memory/media_alt.yaml`) is unconscious and neutral: "Describe what the image shows", in one or two sentences, with the readable text of the image at the end.
Empty, too long (`ALT_TEXT_MAX_CHARS`), or tagged output gets a repair turn.
Provider errors repeat the call. After `ALT_TEXT_ATTEMPTS` calls, the last error propagates.

[`MemoryStore.annotate_alt_text`][pm.memory.store.MemoryStore.annotate_alt_text] stores the result with `alt_text_source="model"` in every record that holds the image, and saves the checkpoint.
The alt text is derived data, like an embedding, so the record revision stays the same.
Open write batches stay valid. A later replacement from an older view keeps the newer alt text (`merge_alt_text` in `commit`).

### Where it runs

Records are embedded lazily on several paths: retrieval, context planning, search, reconsolidation, and the index preparation of the CLI.
All of them call [`EmbeddingService.embed`][pm.memory.embeddings.EmbeddingService.embed].
When the text holds tags, `embed` first asks the installed lookup to complete the missing alt texts.
Then it embeds [`embedding_text`][pm.memory.embeddings.embedding_text]: the text with each tag replaced by `[image: <alt text>]`.
`inspect_index` and `search` use the same text form, so the cache key matches.
The base64 data is never embedded and never enters a text prompt.

For a web chat image, the first main-cognition retrieval embeds the user message, so the alt text exists before the reply is written.

## Prompt encoding

[`encode_messages`][pm.llm.multimodal.encode_messages] is the one place that turns tags into OpenAI-compatible input.
`ChatClient.complete` and context measurement call it. Every workflow that uses the stage model goes through it: main-cognition phases, codelets, dialogue, the assistant, the private loop, and the alt-text call.

| Tag | Result |
| --- | --- |
| First tag of one of the last `PROMPT_MAX_IMAGES` (8) distinct images, in a user turn | An `image_url` part with a `data:` URL, at the position of the tag |
| The same, in a system, assistant, or tool turn | The tag text stays as a reference. The image part goes to the end of the nearest earlier user turn, after `[image:<id>] (shown later in this conversation):` |
| The same, with no earlier user turn | `[image: <alt text>]` |
| A later tag of an image that the prompt already shows | `[image: the same image as shown earlier]` |
| A tag of an older image | `[image: <alt text>]` |
| No alt text yet | `[image: no description yet]` |

The count runs from the end of the last message backwards, across all messages, and counts each image once.
Main-cognition calls repeat the new stimulus in the story, the JSON context, and the instruction.
The image is sent once, in the story prefix, so that prefix keeps the same bytes in every call of a cycle.
A message without tags keeps its plain string content, so a text-only prompt has the same bytes as before.
The OpenAI chat API accepts images only in user turns. The dialogue workflow puts its transcript into an assistant prefill, so its images move to the context turn before it.

Context measurement adds `PROMPT_IMAGE_TOKENS` for each image part. The estimate label is `estimated.images_by_constant.<encoding>`.
The prompt log shows an image part as `[image_url attachment; see request JSON]`. The JSON file beside it has the full request.

### Agent sessions

pm-coder sessions (agent codelets, the Minecraft actor) send their prompts through the pm-coder model client, not through `ChatClient`.
The agent-codelet prompt therefore replaces each tag with its alt text (`codelet_agent.prompt_text`).
pm-coder has its own image path: its `read` tool attaches image files, and MCP tool results can return image blocks.
It keeps the newest images with its own watermark rule (`IMAGE_HIGH_WATER` 32, `IMAGE_LOW_WATER` 8).
That history is separate from record prompts, so the two rules do not interact.

## Memory consolidation

Reconsolidation (summaries, facts, narratives) renders every tag as alt text, never as an image.
[`feature_line`][pm.memory.reconsolidation.story.feature_line] resolves tags through `store.find_image`.
`reconsolidate` first runs `complete_alt_text` for every image in the story features, so no "no description yet" enters memory.
A user's cat photo then reads: `Jesse says: "what is this? [image: A grey cat ...]"`.
The artifact workflow `memory_consolidation` creates records from run files and calls no model. Its image tags get alt text when the records are embedded.

## Web chat images

The composer has an attach button, paste from the clipboard, and drag-and-drop.
Each image shows a preview chip with a remove button. A message can hold up to `WEB_CHAT_MAX_IMAGES` images and no text.
`POST /api/messages` carries the images as [`PostImage`][pm.app.web_chat.PostImage] (`media_type`, `data_b64`, `name`).
`UserInputProxy.submit(text, speaker=..., images=[...])` attaches them to the Dialogue feature and appends the tags.
See [web chat](web-chat.md) for the history and media routes.

## Minecraft screenshots

Before this change, screenshots went this way:

1. The pm-minecraft MCP server returns image blocks. `mcp_client` extracts them as base64.
2. The body gateway writes each image to `screenshots/call-<seq>-<n>.<ext>` in the run workspace. The actor journal (`scratch/actor-journal.jsonl`) names the tool and the paths.
3. The live actor gets only paths and metadata (`BodyGateway(visual=False)`): it uses text evidence. Pixels reach a pm-coder model only with `visual=True`, through MCP image blocks.
4. The artifact consolidator read the files (`image_inventory`) and put base64 into `BodyEpisodeRecord.evidence["images"]`. No prompt and no embedding read it.

Now step 4 uses the shared record media:

- The body episode keeps every screenshot of the run as a `RecordImage`, with `provenance.source="tool"`, `server=MINECRAFT_MCP_SERVER`, the tool name from the actor journal, the run id, and the artifact path.
- Its content ends with the tags of the newest `BODY_EPISODE_TAGGED_IMAGES` screenshots. Only tagged images get alt text and prompt parts.
- `evidence["images"]` keeps the links to the files (path, digest, MIME type, `image_id`), without base64.
- The magic bytes set the MIME type, so a WebP file that the gateway named `.jpg` stays `image/webp`.

Steps 1 to 3 are unchanged. The live actor path keeps its own behavior.

## Limits

- One process holds one media lookup. A second `build_app` in the same process replaces it.
- `find_image` scans the records that hold media. It runs only for tags, but a store with many images makes it slower.
- The alt text of a new image costs one model call, at its first embedding. A provider failure there makes retrieval fall back to lexical ranking for that call.
- Image token counts are a constant estimate. Provider usage after inference is authoritative.
- Audio and video are not record media yet.
