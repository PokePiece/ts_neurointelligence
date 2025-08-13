import { AutoTokenizer, AutoModelForCausalLM } from '@xenova/transformers';
import { ChromaClient, Settings } from 'chromadb/dist/main.js';
import { SentenceTransformer } from '@xenova/transformers';
import * as readline from 'readline/promises';
import { promises as fs } from 'fs';
//import * as process from 'process';
import 'dotenv/config';

const HF_TOKEN = process.env.HUGGING_FACE_TOKEN;

if (!HF_TOKEN) {
    console.error("Hugging Face token not found. Please set HUGGING_FACE_TOKEN in your .env file.");
    process.env.exit(1);
}

// NOTE: This is a placeholder for a Python-only library
function simulateEeg(): number[] {
  // A simple dummy array to represent EEG data
  return Array.from({ length: 2560 }, () => Math.random() * 100);
}

// Global variables for models and ChromaDB client
let tokenizer: AutoTokenizer;
let model: AutoModelForCausalLM;
let embeddingModel: SentenceTransformer;
let client: ChromaClient;
let collection: any;

interface Profile {
    tone: string;
    style: string;
}

/**
 * Initializes and loads all required models and the ChromaDB client.
 */
async function initializeModelsAndDB() {
    try {
        console.log("Loading models and initializing database...");

        // Set up the ChromaDB client
        const settings: Settings = {
            chroma_db_impl: "duckdb+parquet",
            persist_directory: "./chroma_db",
        };
        client = new ChromaClient(settings);
        collection = await client.getOrCreateCollection({ name: "neuro_sessions" });

        // Load the embedding model using Xenova/transformers
        embeddingModel = await SentenceTransformer.from_pretrained('Xenova/all-MiniLM-L6-v2');

        // Load the causal language model
        tokenizer = await AutoTokenizer.from_pretrained("EleutherAI/pythia-160m");
        model = await AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m");
        
        console.log("Models and database initialized successfully.");
    } catch (e) {
        console.error(`An error occurred during initialization: ${e}`);
        process.exit(1);
    }
}

/**
 * Generates text based on a prompt and profile using the Pythia model.
 * @param {string} prompt The text prompt for the model.
 * @param {Profile} profile The profile to influence the model's tone and style.
 * @returns {Promise<string>} The generated text response.
 */
async function generate(prompt: string, profile: Profile): Promise<string> {
    const systemPrompt = `You're a ${profile.tone} assistant. Be ${profile.style}.`;
    const fullPrompt = systemPrompt + "\nUser: " + prompt + "\nAI:";
    
    const inputs = tokenizer(fullPrompt, { return_tensors: "pt" });

    const outputs = await model.generate(inputs, {
        max_new_tokens: 500,
        do_sample: true,
        top_p: 0.9,
        temperature: 0.2,
        repetition_penalty: 1.2,
    });

    const generatedTokens = outputs[0].slice(inputs.input_ids.size[1]);
    return tokenizer.decode(generatedTokens, { skip_special_tokens: true });
}

/**
 * Calls the generate function with a specific neurophysiological prompt.
 * @param {string} eeg The EEG data as a string.
 * @returns {Promise<string>} The analysis from the model.
 */
async function call_generate(eeg: string): Promise<string> {
    console.log('Calling generate');
    const prompt = (
        'You are a neurophysiological data analyzer. Given a set of neurological data, ' +
        'decipher it and report its status. Data: ' + eeg
    );
    const profile = { tone: "technical", style: "analytical" };
    return generate(prompt, profile);
}

/**
 * Extracts a sentence embedding from a signal using the embedding model.
 * @param {number[]} signal The EEG signal data.
 * @returns {Promise<number[]>} The embedding vector.
 */
async function extractEmbedding(signal: number[]): Promise<number[]> {
    const featureText = signal.slice(0, 512).join(" ");
    const embeddings = await embeddingModel.encode(featureText, {
        pooling: 'mean',
        normalize: true,
    });
    return embeddings.data as number[];
}

/**
 * Stores a session in the ChromaDB collection.
 * @param {string} sessionId The unique ID for the session.
 * @param {number[]} embedding The embedding vector for the session.
 * @param {string} note A human-readable note.
 */
async function storeSession(sessionId: string, embedding: number[], note: string): Promise<void> {
    await collection.add({
        ids: [sessionId],
        embeddings: [embedding],
        metadatas: [{ note: note }],
        documents: [note]
    });
}

/**
 * Searches for sessions in the ChromaDB collection.
 * @param {string} query The search query.
 * @param {number} topK The number of top results to return.
 * @returns {Promise<any>} The search results.
 */
async function searchSessions(query: string, topK: number = 3): Promise<any> {
    const queryEmbedding = await embeddingModel.encode(query, {
        pooling: 'mean',
        normalize: true,
    });
    
    const results = await collection.query({
        query_embeddings: [queryEmbedding.data as number[]],
        n_results: topK
    });
    return results;
}

/**
 * Main application loop.
 */
async function main() {
    await initializeModelsAndDB();

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    while (true) {
        const cmd = (await rl.question("Command (store/search/exit): ")).trim().toLowerCase();

        if (cmd === "store") {
            console.log("Simulating EEG...");
            const signal = simulateEeg();
            console.log(`Signal reading: ${signal.slice(0, 5).join(", ")}...`);

            console.log("Extracting embedding...");
            const embedding = await extractEmbedding(signal);

            const note = await rl.question("Enter a note for this session: ");
            const sessionId = await rl.question("Enter a session ID: ");
            console.log(`Embedding reading: ${embedding.slice(0, 5).join(", ")}...`);

            const analysis = await call_generate(embedding.slice(0, 80).join(" "));
            console.log("--- Analysis ---");
            console.log(analysis);

            console.log("Storing session...");
            await storeSession(sessionId, embedding, note);
            console.log("Session stored.\n");

        } else if (cmd === "search") {
            const query = await rl.question("Enter search query: ");
            console.log("Searching...\n");
            const results = await searchSessions(query);

            for (let i = 0; i < results.documents[0].length; i++) {
                const doc = results.documents[0][i];
                const meta = results.metadatas[0][i];
                console.log(`${i + 1}. Note: ${doc}\n  Metadata: ${JSON.stringify(meta)}\n`);
            }

        } else if (cmd === "exit") {
            rl.close();
            break;
        } else {
            console.log("Unknown command.\n");
        }
    }
}

main();
