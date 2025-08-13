// chat.js
import ort from 'onnxruntime-node';
import * as readline from 'readline/promises';
import fs from 'fs';
import { Tokenizer } from "tokenizers"; // Raw tokenizers package



//change to my huggingface model
const modelPath = './exported_model/model.onnx';

let session;

/**
 * Load the ONNX model from local disk
 */
async function loadModel() {
    try {
        console.log("Loading ONNX model...");
        session = await ort.InferenceSession.create(modelPath);
        console.log("Model loaded successfully.");
    } catch (e) {
        console.error("Error loading model:", e);
        process.exit(1);
    }
}

let tokenizer;

import path from 'path';

// ...

async function loadTokenizer() {
  tokenizer = await Tokenizer.fromFile(
    path.resolve("./exported_model/tokenizer.json")
  );
}




/**
 * Run a single inference (stub logic – needs tokenizer and proper input formatting)
 */
async function generate(prompt) {
    const encoded = await tokenizer.encode(prompt);
    const tokens = encoded.ids;
    const attention = new Array(tokens.length).fill(1);
    const position = tokens.map((_, i) => i);

    const inputTensor = new ort.Tensor('int64', BigInt64Array.from(tokens.map(BigInt)), [1, tokens.length]);
    const maskTensor = new ort.Tensor('int64', BigInt64Array.from(attention.map(BigInt)), [1, tokens.length]);
    const positionTensor = new ort.Tensor('int64', BigInt64Array.from(position.map(BigInt)), [1, tokens.length]);

    const feeds = {
        input_ids: inputTensor,
        attention_mask: maskTensor,
        position_ids: positionTensor,
    };

    const results = await session.run(feeds);
    const output = results.logits;

    // Step 1: Get logits for last token
    const vocabSize = output.dims[2]; // 50304 for Pythia
    const lastLogits = output.data.slice(-vocabSize);

    // Step 2: Greedy decoding — find max index
    let max = -Infinity;
    let maxIndex = -1;
    for (let i = 0; i < lastLogits.length; i++) {
        if (lastLogits[i] > max) {
            max = lastLogits[i];
            maxIndex = i;
        }
    }

    // Step 3: Decode token
    const decoded = await tokenizer.decode([maxIndex], { skipSpecialTokens: true });
    return decoded;
}





/**
 * Start the REPL loop
 */
async function main() {
    await loadModel();
    await loadTokenizer();


    console.log("Welcome to the ONNX assistant. Type 'quit' to exit.");

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

        const response = await generate(prompt);
        console.log("AI:", response);
    }
}

main();
