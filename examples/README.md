# Examples

This folder holds **input JSON files** describing the editing tasks that
`ADE_CoT_demo.py` should process, along with their corresponding source
images and (optional) edit-region masks.

## Input JSON schema

```json
{
    "<absolute_or_relative_path_to_input_image>": {
        "instruction":                "<edit instruction in natural language>",

        // Optional — only required when --prune_score_way includes `caption`
        "original_caption":           "<a caption describing the original image>",
        "edited_caption":             "<a caption describing the edited image>",

        // Optional — only required when --prune_score_way includes `region`
        "mask_path":                  "<path to the binary mask of the edit region>",

        // Optional — pre-generated CoT yes/no checklist.
        // If missing, ADE-CoT will auto-generate it via the MLLM verifier
        // and cache it back into the JSON.
        "instance_specific_questions": [
            "Question 1 (yes/no, answer 'yes' iff the edit succeeded)",
            "...",
            "Question 5"
        ]
    }
}
```

## Bundled example

This folder ships a single ready-to-run example, `demo.json`, which edits the
bundled image `case1.png`:

```json
{
    "examples/case1.png": {
        "instruction": "Add a cherry eating action.",
        "original_caption": "A character standing with empty hands.",
        "edited_caption":   "A character eating a cherry."
    }
}
```

## Running the demo

```bash
torchrun --nproc_per_node=1 ../ADE_CoT_demo.py \
    --input_json_dir   ./demo.json \
    --output_dir       ../output \
    --model_name       step1x_edit \
    --model_path       /path/to/Step1X-Edit \
    --num_samples      32
```

> `demo.json` references the bundled `case1.png`, so it runs out of the box.
> To try your own cases, drop new images here and update the paths inside the
> JSON accordingly.
