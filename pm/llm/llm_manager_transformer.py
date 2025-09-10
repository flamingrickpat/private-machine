
class LlmManagerTransformer(LlmManager):
    def __init__(self, test_mode=False):
        super().__init__(test_mode)

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.current_model_id: Optional[str] = None

        # Qwen specific token IDs - get dynamically if possible
        self._think_token_id = 151668  # </think> for Qwen/Qwen3
        self._enable_thinking_flag = False  # Track if thinking is enabled for the current model

    def _get_token_id(self, text: str, add_special_tokens=False) -> Optional[int]:
        """Helper to get the last token ID of a string."""
        if not self.tokenizer:
            return None
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return tokens[-1] if tokens else None
        except Exception:
            return None

    def load_model(self, preset: LlmPreset):
        if preset == LlmPreset.CurrentOne:
            return

        if preset.value not in model_map:
            raise ValueError(f"Unknown model preset: {preset.value}")

        model_config = model_map[preset.value]
        model_hf_id = model_config["path"]
        trust_remote = False

        if self.current_model_id != model_hf_id:
            logging.info(f"Loading model: {model_hf_id}...")
            # Release old model resources
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(1)

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_hf_id,
                    trust_remote_code=trust_remote,
                    local_files_only=local_files_only
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_hf_id,
                    torch_dtype="auto",  # Use appropriate dtype (bfloat16 often good)
                    device_map="auto",  # Automatically distribute across available GPUs/CPU
                    trust_remote_code=trust_remote,
                    attn_implementation="flash_attention_2", # if supported and desired,
                    local_files_only=local_files_only,
                    load_in_8bit=True,
                    use_safetensors=True
                )
                self.current_model_id = model_hf_id

                # Special handling for Qwen thinking tag based on model ID
                if "qwen" in model_hf_id.lower():
                    # Try to get </think> token ID dynamically
                    qwen_think_id = self._get_token_id("</think>", add_special_tokens=False)
                    if qwen_think_id:
                        self._think_token_id = qwen_think_id
                        logging.info(f"Detected Qwen model. Using </think> token ID: {self._think_token_id}")
                    else:
                        logging.info(f"WARN: Could not dynamically get </think> token ID for {model_hf_id}. Using default: {self._think_token_id}")

                logging.info(f"Model {model_hf_id} loaded successfully.")

            except Exception as e:
                logging.info(f"Error loading model {model_hf_id}: {e}")
                self.current_model_id = None
                self.model = None
                self.tokenizer = None
                raise  # Re-raise the exception

    def completion_tool(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: Optional[CommonCompSettings] = None,
                        tools: Optional[List[Type[BaseModel]]] = None,
                        discard_thinks: bool = True
                        ) -> Tuple[str, List[BaseModel]]:

        # return controller.completion_tool(preset, inp, comp_settings, tools)

        self.load_model(preset)
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded successfully.")

        # Ensure comp_settings exists
        if comp_settings is None:
            comp_settings = CommonCompSettings()
        if tools is None:
            tools = []

        # --- Input Formatting ---
        inp_formatted = []
        for msg in inp:
            inp_formatted.append((self.format_str(msg[0]), self.format_str(msg[1])))

        # Merge consecutive messages from the same role
        merged_inp_formatted = []
        if inp_formatted:
            current_role, current_content = inp_formatted[0]
            for i in range(1, len(inp_formatted)):
                role, content = inp_formatted[i]
                if role == current_role:
                    current_content += "\n" + content
                else:
                    merged_inp_formatted.append((current_role, current_content))
                    current_role, current_content = role, content
            merged_inp_formatted.append((current_role, current_content))
        inp_formatted = merged_inp_formatted

        # --- Tool Setup ---
        tool_schema_str = ""
        if len(tools) > 0:
            # Assuming only one tool for now, as in the original code
            tool_model = tools[0]
            try:
                schema = tool_model.model_json_schema()
                tool_schema_str = f"\nYou have access to a tool. Use it by outputting JSON conforming to this schema:\n{json.dumps(schema)}"
                # Prepend schema to the system prompt (if one exists) or add a new system message
                if inp_formatted and inp_formatted[0][0] == "system":
                    inp_formatted[0] = ("system", inp_formatted[0][1] + tool_schema_str)
                else:
                    # Add schema as the first system message
                    inp_formatted.insert(0, ("system", tool_schema_str.strip()))
            except Exception as e:
                logging.info(f"Warning: Could not generate tool schema: {e}")
                tools = []  # Disable tools if schema fails

        # --- Logging Setup ---
        log_filename = f"{get_call_stack_str()}_{uuid.uuid4()}.log"
        os.makedirs(log_path, exist_ok=True)
        log_file_path = os.path.join(log_path, log_filename)
        log_conversation(inp_formatted, log_file_path, 200)

        # --- Prepare for Tokenization and Context Trimming ---
        openai_inp = [{"role": msg[0], "content": msg[1]} for msg in inp_formatted]

        # Determine max context length
        # Use model's config, fallback to a reasonable default
        max_context_len = getattr(self.model.config, 'max_position_embeddings', 4096)
        # Reserve space for generation and a buffer
        max_prompt_len = max_context_len - comp_settings.max_tokens - 32  # 32 tokens buffer

        # Apply chat template - IMPORTANT: handle thinking flag
        template_args = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        # Conditionally add enable_thinking for models that support it (like Qwen)
        if self._enable_thinking_flag:
            template_args["enable_thinking"] = True  # Assuming this is how Qwen expects it

        template_args["enable_thinking"] = False

        # check if seedint promtp is necessary
        probably_mistral = False
        try:
            # remove add generation prompts if last turn is from assistant
            openai_inp_remove_ass = [{"role": "system", "content": "test1"}, {"role": "user", "content": "test2"}, {"role": "assistant", "content": "test3"}, {"role": "assistant", "content": "test4"}]
            data_remove_ass = self.tokenizer.apply_chat_template(openai_inp_remove_ass, **template_args)
        except:
            # remove add generation prompts if last turn is from assistant
            openai_inp_remove_ass = [{"role": "system", "content": "test1"}, {"role": "user", "content": "test2"}, {"role": "assistant", "content": "test4"}]
            data_remove_ass = self.tokenizer.apply_chat_template(openai_inp_remove_ass, **template_args)
            probably_mistral = True

        remove_ass_string = data_remove_ass.split("test4")[1]

        # --- Context Trimming Loop ---
        while True:
            try:
                prompt_text = self.tokenizer.apply_chat_template(openai_inp, **template_args)
            except Exception as e:
                # Fallback if apply_chat_template fails (e.g., template not found)
                logging.info(f"Warning: apply_chat_template failed: {e}. Using basic concatenation.")
                prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in openai_inp]) + "\nassistant:"  # Simple fallback

            if openai_inp[-1]["role"] == "assistant":
                prompt_text = prompt_text.removesuffix(remove_ass_string)

            # Tokenize to check length
            # Note: We tokenize *without* adding special tokens here, as apply_chat_template usually handles them.
            # If your template doesn't, adjust accordingly.
            tokenized_prompt = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)  # Let template handle BOS/EOS
            prompt_token_count = tokenized_prompt.input_ids.shape[1]

            if prompt_token_count > max_prompt_len:
                if len(openai_inp) <= 1:  # Cannot remove more messages
                    logging.info(f"Warning: Input prompt ({prompt_token_count} tokens) is too long even after removing messages, exceeding max_prompt_len ({max_prompt_len}). Truncating.")
                    # Truncate the tokenized input directly (might break formatting)
                    tokenized_prompt.input_ids = tokenized_prompt.input_ids[:, -max_prompt_len:]
                    tokenized_prompt.attention_mask = tokenized_prompt.attention_mask[:, -max_prompt_len:]
                    break
                # Remove message from the middle (excluding system prompt if present)
                remove_index = 1 if openai_inp[0]["role"] == "system" and len(openai_inp) > 2 else 0
                remove_index = max(remove_index, (len(openai_inp) // 2))  # Try middle, but don't remove system prompt easily
                logging.info(f"Context too long ({prompt_token_count} > {max_prompt_len}). Removing message at index {remove_index}: {openai_inp[remove_index]['role']}")
                del openai_inp[remove_index]
            else:
                break  # Prompt fits

        model_inputs = tokenized_prompt.to(self.model.device)
        input_length = model_inputs.input_ids.shape[1]  # Store for separating output later

        # class MinPLogitsWarper(LogitsWarper):
        #    def __init__(self, min_p): self.min_p = min_p
        #    def __call__(self, input_ids, scores):
        #        probs = torch.softmax(scores, dim=-1)
        #        mask  = probs < self.min_p
        #        scores = scores.masked_fill(mask, -1e9)
        #        return scores


        # --- Generation Arguments ---
        warpers = LogitsProcessorList([])

        if False:
            warpers.append(TemperatureLogitsWarper(temperature=0.7))
            warpers.append(TopPLogitsWarper(top_p=0.8))
            warpers.append(TopKLogitsWarper(top_k=20))
            warpers.append(MinPLogitsWarper(min_p=0.0))

        if False and len(tools) == 0:
            # Instantiate the custom logits processor
            if self.model.config.eos_token_id is None:
                # Fallback for models that might not explicitly set eos_token_id in config
                # but tokenizer has it (e.g., some older GPT-2 checkpoints)
                eos_token_id = self.tokenizer.eos_token_id
            else:
                eos_token_id = self.model.config.eos_token_id

            target_length_processor = TargetLengthLogitsProcessor(
                prompt_len=input_length,
                target_output_len=48,
                eos_token_id=eos_token_id,
                tokenizer=self.tokenizer,
                penalty_window=4,  # Start penalizing/boosting EOS +/- 2 tokens around target
                base_eos_boost=15.0,  # A significant boost to make EOS likely
                overshoot_penalty_factor=2.0,  # Make it even more likely to stop if over target
                sentence_end_chars=".!?。",
                sentence_end_boost_factor=2.5,  # Stronger preference for EOS after sentence end
                min_len_to_apply=2  # Don't apply for the first 2 generated tokens
            )
            warpers.append(target_length_processor)

        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": comp_settings.max_tokens,
            "repetition_penalty": 1.2,  # comp_settings.repeat_penalty,
            "temperature": 0.7, #comp_settings.temperature if comp_settings.temperature > 1e-6 else 1.0,  # Temp 0 means greedy, set to 1.0
            "do_sample": True, # comp_settings.temperature > 1e-6,  # Enable sampling if temp > 0
            "top_k": 20,
            "top_p": 0.8,
            "min_p": 0,
            # "frequency_penalty": 1.05, # comp_settings.frequency_penalty,
            # "presence_penalty": 1, # comp_settings.presence_penalty,
            "logits_processor": warpers,
            # "pad_token_id": self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id, # Handle padding
            # "eos_token_id": self.tokenizer.eos_token_id
        }

        # Add penalties only if they are non-zero (some models might error otherwise)
        #if comp_settings.frequency_penalty == 0.0: del generation_kwargs["frequency_penalty"]
        #f comp_settings.presence_penalty == 0.0: del generation_kwargs["presence_penalty"]

        # TODO: Mirostat is not standard in transformers.generate. Would require custom sampling logic. Skipping for now.
        # if preset == LlmPreset.Conscious:
        #    logging.info("Warning: Mirostat settings are not directly supported by default transformers.generate.")

        # --- Setup Logits Processors and Stopping Criteria ---
        logits_processor = LogitsProcessorList()
        stopping_criteria = StoppingCriteriaList()
        prefix_function = None

        # 1. LM Format Enforcer (if tools are present)
        if len(tools) > 0:
            if self.test_mode:
                return "empty", [create_random_pydantic_instance(tools[0])]

            try:
                os.environ["LMFE_STRICT_JSON_FIELD_ORDER"] = "1"
                os.environ["LMFE_MAX_JSON_ARRAY_LENGTH"] = "5"
                #parser = JsonSchemaParser(tools[0].model_json_schema())
                #prefix_function = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)

                # prefix_allowed_tokens_fn = lmformatenforcer.SequencePrefixParser(parser)
                # format_processor = lmformatenforcer.hf.LogitsProcessor(prefix_allowed_tokens_fn)
                # logits_processor.append(format_processor)
                logging.info("Added LM Format Enforcer for tool JSON.")
            except Exception as e:
                logging.info(f"Warning: Failed to initialize LM Format Enforcer: {e}. Tool usage might fail.")
                tools = []  # Disable tools if enforcer fails

        # 2. Custom Stop Words
        if comp_settings.stop_words:
            stop_token_ids = []
            for word in comp_settings.stop_words:
                # Encode stop words *without* special tokens, but check behavior
                # Some tokenizers might add spaces, handle carefully.
                # This encodes the word as if it appears mid-sequence.
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if ids:
                    stop_token_ids.append(ids)
            if stop_token_ids:
                stopping_criteria.append(StopOnTokens(stop_token_ids))
                logging.info(f"Added custom stopping criteria for: {comp_settings.stop_words}")

        if len(logits_processor) > 0:
            generation_kwargs["logits_processor"] = logits_processor
        if len(stopping_criteria) > 0:
            generation_kwargs["stopping_criteria"] = stopping_criteria
        if prefix_function:
            generation_kwargs["prefix_allowed_tokens_fn"] = prefix_function

        # --- Generation ---
        logging.info("Starting generation...")
        content = ""
        calls = []
        generated_text = ""
        finish_reason = "unknown"  # Transformers doesn't explicitly return this like OpenAI API
        thinking_content = ""

        try:
            with torch.no_grad():  # Ensure no gradients are computed
                generated_ids = self.model.generate(
                    **model_inputs,
                    **generation_kwargs
                )

            # Extract only the generated tokens
            output_ids = generated_ids[0][input_length:].tolist()

            # --- Post-processing ---

            # 1. Handle Thinking Tags (Qwen specific, adapt if needed)
            content_ids = output_ids  # Default to all output
            if self._enable_thinking_flag and self._think_token_id in output_ids:
                try:
                    # Find the *last* occurrence of the think token ID
                    # rindex_token = len(output_ids) - 1 - output_ids[::-1].index(self._think_token_id) # Find last index
                    # Find the *first* occurrence for Qwen's format
                    index = output_ids.index(self._think_token_id)
                    # Decode thinking part (up to and including </think>)
                    thinking_content = self.tokenizer.decode(output_ids[:index + 1], skip_special_tokens=True).strip()
                    # Get content part (after </think>)
                    content_ids = output_ids[index + 1:]
                    logging.info(f"Separated thinking content (length {len(thinking_content)}).")
                except ValueError:
                    logging.info("Warning: </think> token ID found but failed to split.")
                    # Fallback: keep all as content if splitting fails
                    content_ids = output_ids

            # 2. Decode the final content part
            content = self.tokenizer.decode(content_ids, skip_special_tokens=True).strip()

            # Determine finish reason (simple heuristics)
            if self.tokenizer.eos_token_id in output_ids:
                finish_reason = "stop"
            elif len(output_ids) >= comp_settings.max_tokens - 1:  # -1 for safety
                finish_reason = "length"
            # Check if stopped by custom criteria (harder to check definitively after the fact)
            # We know if StopOnTokens returned True if generation stopped *before* max_tokens or EOS.

            # 3. Handle Tool Calls (if format enforcer was used)
            if len(tools) > 0:
                # The 'content' should now be the JSON string enforced by the processor
                try:
                    # No need for repair_json if enforcer worked
                    validated_model = tool_model.model_validate_json(content)
                    calls = [validated_model]
                    # Optionally clear content if only tool call is expected
                    # content = "" # Or keep it if mixed output is possible
                    logging.info("Successfully parsed tool JSON.")
                except Exception as e:
                    logging.info(f"Error parsing enforced JSON: {e}. Content: '{content}'")
                    # Fallback: return raw content, no validated calls
                    calls = []
            else:
                # Normal text generation (potentially with thinking part removed)
                calls = []
                if discard_thinks and thinking_content:
                    # Content is already the part after </think>
                    pass
                elif not discard_thinks and thinking_content:
                    # Prepend thinking content if not discarding
                    content = f"<think>{thinking_content}</think>\n{content}"  # Reconstruct if needed


        except Exception as e:
            logging.info(f"Error during generation or processing: {e}")
            # Handle error case, maybe return empty or partial results
            content = f"Error during generation: {e}"
            calls = []
            finish_reason = "error"

        # --- Final Logging ---
        final_output_message = ("assistant", f"<think>{thinking_content}</think>\n{content}" if thinking_content else content)  # Log full output before discard
        log_conversation(inp_formatted + [final_output_message], log_file_path + f".completion_{finish_reason}.log", 200)

        # Return final processed content and tool calls
        return content, calls

    def completion_text(self,
                        preset: LlmPreset,
                        inp: List[Tuple[str, str]],
                        comp_settings: Optional[CommonCompSettings] = None,
                        discard_thinks: bool = True) -> str:

        # return controller.completion_text(preset, inp, comp_settings)

        if self.test_mode:
            return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(128))

        # No tools expected in this simplified method
        content, _ = self.completion_tool(preset, inp, comp_settings, tools=None, discard_thinks=discard_thinks)
        return content  # Already stripped in completion_tool

