// chat.js
import { AutoTokenizer, AutoModelForCausalLM } from '@xenova/transformers';
import * as readline from 'readline/promises';
import 'dotenv/config';
import * as process from 'process';

// Get Hugging Face token from environment variables for security.
const HF_TOKEN = process.env.HUGGING_FACE_TOKEN;

// Throw an error if the token is not found, as it's required for some models.
if (!HF_TOKEN) {
    console.error("Hugging Face token not found. Please set HUGGING_FACE_TOKEN in your .env file.");
    process.exit(1);
}

// Define the model name to be used.
const model_name = "xenova/pythia-410m";

let tokenizer;
let model;

/**
 * Initializes and loads the tokenizer and the language model.
 * The '@xenova/transformers' library automatically handles memory optimization,
 * similar to Python's bfloat16, by using a quantized model by default.
 */
async function loadModel() {
    try {
        console.log("Loading model to save memory...");
        tokenizer = await AutoTokenizer.from_pretrained(model_name);
        model = await AutoModelForCausalLM.from_pretrained(model_name, {
            // This is the default behavior to save memory.
            quantized: true
        });
        console.log("Model loaded successfully.");
    } catch (e) {
        console.error(`An error occurred during model loading: ${e}`);
        process.exit(1);
    }
}

/**
 * Interface for the user profile to define the assistant's tone and style.
 * @typedef {Object} Profile
 * @property {string} tone - The tone of the assistant (e.g., "realistic").
 * @property {string} style - The style of the assistant (e.g., "technical").
 */

/**
 * Generates a response from the model based on a user prompt and profile.
 * @param {string} prompt The user's input.
 * @param {Profile} profile The profile to set the assistant's tone and style.
 * @returns {Promise<string>} The generated response from the AI.
 */
async function generate(prompt, profile) {
    const system_prompt = `You're a ${profile.tone} assistant. Be ${profile.style}.`;
    const full_prompt = system_prompt + "\nUser: " + prompt + "\nAI:";

    // Tokenize the full prompt.
    // Fixed:
    const inputs = await tokenizer(full_prompt);
    const outputs = await model.generate(inputs.input_ids, {
        max_new_tokens: 500,
        do_sample: true,
        top_p: 0.9,
        temperature: 0.2,
        repetition_penalty: 1.2,
    });


    // Decode the generated tokens to a readable string.
    const generated_tokens = outputs[0].slice(inputs.input_ids.size[1]);
    return tokenizer.decode(generated_tokens, { skip_special_tokens: true });
}

/**
 * Main function to handle the interactive chat loop.
 */
async function main() {
    await loadModel();

    console.log("Welcome to the interactive AI assistant. Type 'quit' to exit.");

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    while (true) {
        const prompt = await rl.question("You: ");
        if (prompt.toLowerCase() === "quit" || prompt.toLowerCase() === "exit") {
            console.log("Exiting... Goodbye!");
            rl.close();
            break;
        }

        const profile = { "tone": "realistic", "style": "technical" };
        const response = await generate(prompt, profile);
        console.log("AI:", response);
    }
}

// Start the application.
main();
