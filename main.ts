/**
 *  Telegram Autobot. Do not use.
 *  -----------------------------------
 */

import { Api, TelegramClient } from 'telegram';
import { StringSession } from 'telegram/sessions/index.js';
import { NewMessage } from 'telegram/events/index.js';
import * as dotenv from 'dotenv';
import input from 'input';
import { exec, spawn } from 'child_process';
import axios from 'axios';
import { promises as fs } from 'fs';
import { CustomFile } from 'telegram/client/uploads.js';
import OpenAI from 'openai';
import { zodResponseFormat } from 'openai/helpers/zod';
import { z } from 'zod';

dotenv.config();

/* ------------------------------------------------------------------
   Environment Variables
   ------------------------------------------------------------------ */

const {
  TG_API_ID,
  TG_API_HASH,
  PHONE_NUMBER,
  SESSION_NAME,
  MY_USERNAME,
  LLM_CONTEXT_PATH,
  COMMAND_BASE_URL,
  LLMM1,
  OZZO_API_KEY,
  OZZO_GETCONT,
  OPENAI_API_KEY,
} = process.env;

const AUTHORIZED_IDS = [
 '<RETARCTED>'
];

const UNAUTHORIZED_IDS = [
 '<RETARCTED>'
]

/** Verify required environment variables */
if (
  !TG_API_ID ||
  !TG_API_HASH ||
  !PHONE_NUMBER ||
  !OPENAI_API_KEY ||
  !MY_USERNAME ||
  !LLM_CONTEXT_PATH ||
  !COMMAND_BASE_URL
) {
  console.error(
    'Missing required environment variables. Please check your .env file.'
  );
  process.exit(1);
}

/* ------------------------------------------------------------------
   Constants & Initialization
   ------------------------------------------------------------------ */

const LOGS_DIR = './logs'; // Directory where conversation logs are stored
const MAX_UNIQUE_LINES = 500; // For the /z command compressed-file search
const MAX_LOGIN_ATTEMPTS = 3; // Max attempts to log in to Telegram
const SAVE_INTERVAL_MS = 5 * 60 * 1000; // Save in-memory messages every 5 minutes

// Create the Telegram client using a StringSession
const stringSession = new StringSession(SESSION_NAME);
const client = new TelegramClient(
  stringSession,
  parseInt(TG_API_ID, 10),
  TG_API_HASH,
  { connectionRetries: 5 }
);

/** Maps a log filename to an in-memory array of messages not yet saved to file. */
const messagesMap = {};

/** Caches { entityId: entityNameOrId } to avoid repeated entity lookups */
const entityCache = {};

/** OpenAI clients */
const openai = new OpenAI({
  apiKey: OPENAI_API_KEY,
});

const uncensoredai = new OpenAI({ baseURL: 'http://192.168.1.151:1234' });

/** Commands structure definition */
const COMMANDS = {
  q: {
    prefix: '!q',
    endpoint: `${COMMAND_BASE_URL}/q`,
    params: {
      temperature: 0.3,
      tokens: 3000,
      model: LLMM1,
      version: 'v2',
    },
  },
  b: {
    prefix: '!cb',
    endpoint: `${COMMAND_BASE_URL}/q`,
    params: {
      temperature: 0.92,
      tokens: 4000,
      format: 'fun',
      model: 'openai-next',
      version: 'v2',
    },
  },
  w: {
    prefix: '!w',
    endpoint: OZZO_GETCONT,
    params: {
      headers: {
        'X-API-KEY': OZZO_API_KEY,
        'Content-Type': 'application/json',
      },
    },
  },
  arf: { prefix: '!arf' },
  summarize: { prefix: '!cai' },
  search: { prefix: '!search' },
  stats: { prefix: '!stats' },
  fsearch: { prefix: '!fsearch' },
  z: { prefix: '/z' },
};

/* ------------------------------------------------------------------
   Utility Functions
   ------------------------------------------------------------------ */

/**
 * Sleep helper for async/await.
 * @param {number} ms
 * @returns {Promise<void>}
 */
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Checks if a given user ID is in the authorized list.
 * @param {string | number} userId
 * @returns {boolean}
 */
function isUserAuthorized(userId) {
  // Convert to string if needed
  const idStr = String(userId);
  return AUTHORIZED_IDS.includes(idStr);
}

/**
 * Slugifies a text string (lowercase, remove special chars, etc.).
 * @param {string} text
 * @returns {string}
 */
function slugify(text) {
  return text
    .normalize('NFD') // Normalize to decompose accents
    .replace(/[\u0300-\u036f]/g, '') // Remove accents
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9 -]/g, '') // Remove all non-alphanumeric chars except spaces/hyphens
    .replace(/\s+/g, '-')
    .replace(/-+/g, '-');
}

/**
 * Returns the ratio of matched characters to query length.
 * @param {string} text
 * @param {string} query
 * @returns {number}
 */
function fuzzyScore(text, query) {
  const t = text.toLowerCase();
  const q = query.toLowerCase();

  let tIndex = 0;
  let qIndex = 0;
  let matches = 0;

  while (tIndex < t.length && qIndex < q.length) {
    if (t[tIndex] === q[qIndex]) {
      matches++;
      qIndex++;
    }
    tIndex++;
  }

  return matches / q.length;
}

/**
 * Parses text to see if it matches any known command prefix.
 * @param {string} text
 * @returns {{ command: string, content: string } | null}
 */
function parseCommand(text) {
  if (!text) return null;
  for (const [key, value] of Object.entries(COMMANDS)) {
    if (text.startsWith(value.prefix)) {
      return { command: key, content: text.slice(value.prefix.length).trim() };
    }
  }
  return null;
}

/**
 * POST to the provided endpoint with content/context. Returns string or object.
 * @param {string} command
 * @param {string} content
 * @param {string} [context='']
 * @returns {Promise<string | { name: string, data: Buffer, text: string, json?: any }>}
 */
async function sendPromptAndGetResponse(command, content, context = '') {
  const { endpoint, params } = COMMANDS[command];
  try {
    if (command === 'w') {
      // Wizard command (example)
      const response = await axios.post(endpoint, { url: content }, {
        headers: params.headers,
      });
      return {
        name: slugify(content),
        data: Buffer.from(response.data.markdown, 'utf-8'),
        text: response.data.markdown,
      };
    }

    // Generic LLM endpoint
    const response = await axios.post(
      endpoint,
      { question: content, context, ...params },
      { headers: { 'Content-Type': 'application/json' } }
    );
    return response.data.assistant;
  } catch (error) {
    console.error(`Error sending prompt to ${endpoint}:`, error);
    return 'Sorry, I encountered an error while processing your request.';
  }
}

/**
 * Converts markdown text into a PDF buffer.
 * @param {string} markdown
 * @returns {Promise<Buffer>}
 */
async function convertMarkdownToPdf(markdown) {
  const params = new URLSearchParams();
  params.append('markdown', markdown);

  const response = await axios.post('https://md-to-pdf.fly.dev', params, {
    responseType: 'arraybuffer',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  });
  return Buffer.from(response.data);
}

/* ------------------------------------------------------------------
   OpenAI-based Summaries
   ------------------------------------------------------------------ */

/** For demonstration: We define a Zod schema for a summary. */
const SummarySchema = z.object({ summary: z.string() });

/**
 * Summarize a chat log with the official OpenAI instance.
 * @param {string} chatText
 * @returns {Promise<string>}
 */
async function getChatSummary(chatText) {
  const completion = await openai.beta.chat.completions.parse({
    model: 'gpt-4o-2024-08-06',
    messages: [
      { role: 'system', content: 'Provide a concise summary of the following chat.' },
      { role: 'user', content: chatText },
    ],
    response_format: zodResponseFormat(SummarySchema, 'summary'),
    max_tokens: 200,
    temperature: 0.3,
  });

  // Parsed content matches our schema: { summary: string }
  const parsed = completion.choices[0].message.parsed;
  return parsed.summary.trim();
}

/**
 * Summarize a chat with an "uncensored" model (local or alternative endpoint).
 * @param {string} chatText
 * @returns {Promise<string>}
 */
async function getChatSummaryUncensored(chatText) {
  const data = JSON.stringify({
    model:
      'solar-10.7b-instruct-v1.0-uncensored',
    messages: [
      {
        role: 'system',
        content:
          `You're an FBi agent who is a mole inside a private or group chat. When prompted, write a detailed, elaborate, fine-grained summary of the provided messages, as if you were reporting to your superiors.`,
      },
      {
        role: 'user',
        content: chatText,
      },
    ],
    max_tokens: 800,
    temperature: 0.6,
    stream: false,
  });

  const config = {
    method: 'post',
    maxBodyLength: Infinity,
    url: 'http://0.0.0.0:1234/v1/chat/completions',
    headers: { 'Content-Type': 'application/json' },
    data,
  };

  const response = await axios.request(config);
  return response.data.choices[0].message.content;
}

/* ------------------------------------------------------------------
   Logging Helpers
   ------------------------------------------------------------------ */

/**
 * Reads a JSON log file, returning an array of messages (or empty array).
 * @param {string} filepath
 * @returns {Promise<any[]>}
 */
async function readLogFile(filepath) {
  try {
    const fileData = await fs.readFile(filepath, 'utf8');
    return JSON.parse(fileData);
  } catch {
    return [];
  }
}

/**
 * Writes an array of messages to the specified JSON file.
 * @param {string} filepath
 * @param {Array<any>} messages
 * @returns {Promise<void>}
 */
async function writeLogFile(filepath, messages) {
  const dataStr = JSON.stringify(
    messages,
    (key, value) => (typeof value === 'bigint' ? value.toString() : value),
    2
  );
  await fs.writeFile(filepath, dataStr, 'utf-8');
}

/**
 * Periodically saves any unsaved messages in memory to log files.
 */
async function saveMessagesToFile() {
  for (const filename in messagesMap) {
    const unsaved = messagesMap[filename];
    if (unsaved.length > 0) {
      const filepath = `${LOGS_DIR}/${filename}`;
      try {
        const existing = await readLogFile(filepath);
        const merged = existing.concat(unsaved);
        await writeLogFile(filepath, merged);
        messagesMap[filename] = [];
      } catch (error) {
        console.error(`Error writing to file ${filename}:`, error);
      }
    }
  }
}

/* ------------------------------------------------------------------
   Telegram Login
   ------------------------------------------------------------------ */

/**
 * Logs in to Telegram (up to MAX_LOGIN_ATTEMPTS).
 * @returns {Promise<boolean>}
 */
async function login() {
  console.log('Starting login process...');
  for (let attempts = 1; attempts <= MAX_LOGIN_ATTEMPTS; attempts++) {
    try {
      console.log(`Attempt ${attempts} of ${MAX_LOGIN_ATTEMPTS}`);
      await client.start({
        phoneNumber: () => PHONE_NUMBER,
        password: () => input.text('Please enter your 2FA password: '),
        phoneCode: () => input.text('Please enter the code you received: '),
        onError: (err) => console.error(err),
      });
      console.log('Login successful!');
      return true;
    } catch (error) {
      console.error(`Error during login: ${error.message}`);
      if (error.message.includes('PHONE_CODE_INVALID')) {
        console.log('Invalid phone code. Please try again.');
      } else if (error.message.includes('FLOOD_WAIT_')) {
        const waitTime = parseInt(error.message.split('_')[2]);
        console.log(
          `FloodWaitError: Need to wait ${waitTime} seconds before trying again.`
        );
        await sleep(waitTime * 1000);
      } else {
        throw error;
      }
    }
  }
  console.error('Max login attempts reached. Please try again later.');
  return false;
}

/* ------------------------------------------------------------------
   Command Handlers
   ------------------------------------------------------------------ */

/**
 * Handles the "summarize" command (`!cai`) to summarize chat logs from "daysAgo".
 */
async function handleSummarizeCommand(entityId, entityNameOrId, msg) {
  // parsedCommand.content might look like '-d 1' or '-d 2'
  const args = msg.text.split(/\s+/).filter(Boolean).slice(1); // remove the "!cai" portion
  let daysAgo = 0;

  const dIndex = args.indexOf('-d');
  if (dIndex !== -1 && args[dIndex + 1]) {
    const parsedDays = parseInt(args[dIndex + 1], 10);
    daysAgo = isNaN(parsedDays) ? 0 : parsedDays;
  }

  // Compute date offset
  const currentDate = new Date();
  const targetDate = new Date(
    currentDate.getTime() - daysAgo * 24 * 60 * 60 * 1000
  );
  const dateString = targetDate.toISOString().split('T')[0];
  const filenameForSum = `${entityNameOrId}-${dateString}.json`;

  let combinedMessages = [];
  const filePath = `${LOGS_DIR}/${filenameForSum}`;
  // Merged file + in-memory
  const fileMessages = await readLogFile(filePath);
  const memMessages = messagesMap[filenameForSum] || [];
  combinedMessages = fileMessages.concat(memMessages);

  // Last ~200 messages
  const recentCount = 200;
  const recentMessages = combinedMessages.slice(-recentCount);

  // Concatenate for the LLM
  const chatText = recentMessages
    .map(
      (m) =>
        `${m.senderUsername || m.senderFirstName || 'User'}: ${m.text || ''}`
    )
    .join('\n');

  const summary = await getChatSummaryUncensored(chatText);
  await client.sendMessage(entityId, {
    message: `**Summary (${daysAgo} day(s) ago):**\n${summary}`,
    replyTo: msg.id,
  });
}

/**
 * Handles the "stats" command to show simple usage statistics for the day.
 */
/**
 * Handles the "stats" command, showing enhanced usage metrics for today's log.
 */
async function handleStatsCommand(entityId, filename, msg) {
  const filePath = `${LOGS_DIR}/${filename}`;
  const fileMessages = await readLogFile(filePath);
  const memMessages = messagesMap[filename] || [];
  const allMessages = fileMessages.concat(memMessages);

  if (allMessages.length === 0) {
    await client.sendMessage(entityId, {
      message: `No messages logged yet for today.`,
      replyTo: msg.id,
    });
    return;
  }

  // 1) Total Messages
  const totalMessages = allMessages.length;

  // 2) Unique Participants (by username or firstName, fallback to 'User')
  const participantSet = new Set();
  allMessages.forEach(m => {
    const user = m.senderUsername || m.senderFirstName || 'User';
    participantSet.add(user);
  });
  const uniqueParticipants = participantSet.size;

  // 3) Top Participants
  const userCountMap = {};
  allMessages.forEach((m) => {
    const user = m.senderUsername || m.senderFirstName || 'User';
    userCountMap[user] = (userCountMap[user] || 0) + 1;
  });
  const sortedUsers = Object.entries(userCountMap).sort((a, b) => b[1] - a[1]);
  const topUsers = sortedUsers
    .slice(0, 5)
    .map(([user, count]) => `${user}: ${count} messages`)
    .join('\n');

  // 4) Average Message Length
  let totalLength = 0;
  allMessages.forEach((m) => {
    if (m.text) totalLength += m.text.length;
  });
  const avgLength = (totalLength / totalMessages).toFixed(2);

  // 5) Top 10 Words (very naive word splitting)
  const wordFreq = {};
  for (const m of allMessages) {
    if (!m.text) continue;
    // split on whitespace, punctuation-agnostic, etc.
    // remove punctuation or adjust the regex to your liking
    const words = m.text
      .toLowerCase()
      .split(/[\s,!.?;:()]+/)
      .filter(Boolean);

    for (const w of words) {
      wordFreq[w] = (wordFreq[w] || 0) + 1;
    }
  }
  // Sort descending by frequency
  const topWordEntries = Object.entries(wordFreq).sort((a, b) => b[1] - a[1]);
  const topWords = topWordEntries.slice(0, 10).map(([word, count]) => `${word} (${count})`);

  // 6) Hourly Distribution (UTC-based or local-based depending on your preference)
  const hours = Array(24).fill(0);
  allMessages.forEach((m) => {
    if (!m.date) return;
    const dateObj = new Date(m.date);
    const hour = dateObj.getHours(); // local hour; use getUTCHours() for UTC
    hours[hour]++;
  });
  // Prepare a small textual histogram or top hours list
  // For brevity, let's just show the hours with the most messages.
  const hourEntries = hours.map((count, h) => ({ hour: h, count }));
  hourEntries.sort((a, b) => b.count - a.count);
  const topHours = hourEntries
    .slice(0, 3)
    .map(({ hour, count }) => `${hour}:00 - ${count} msg`)
    .join('\n');

  // 7) Top Commands used (naive: check if text starts with "!")
  const commandCounts = {};
  allMessages.forEach((m) => {
    if (!m.text) return;
    const trimmed = m.text.trim();
    if (trimmed.startsWith('!') || trimmed.startsWith('/')) {
      // Extract the first "token" as the command, e.g. "!q", "/z"
      const parts = trimmed.split(/\s+/);
      const cmd = parts[0];
      commandCounts[cmd] = (commandCounts[cmd] || 0) + 1;
    }
  });
  const topCommandEntries = Object.entries(commandCounts).sort((a, b) => b[1] - a[1]);
  const topCommands = topCommandEntries
    .slice(0, 5)
    .map(([cmd, count]) => `${cmd} (${count})`)
    .join(', ');

  // Format message
  const statsMessage = `
**Today's Stats**:
• Total messages: \`${totalMessages}\`
• Unique participants: \`${uniqueParticipants}\`
• Top participants:
${topUsers || 'No messages yet.'}

• Avg message length: \`${avgLength}\` chars
• Top 10 words: ${topWords.join(', ') || 'N/A'}
• Top 3 hours (Local Time):
${topHours || 'N/A'}

• Top 5 commands: ${topCommands || 'N/A'}
`;

  await client.sendMessage(entityId, {
    message: statsMessage.trim(),
    replyTo: msg.id,
  });
}

/**
 * Handles the "search" command to do a direct substring search in today's logs.
 */
async function handleSearchCommand(entityId, filename, query, msg) {
  if (!query) {
    await client.sendMessage(entityId, {
      message: 'Please provide a search keyword.',
      replyTo: msg.id,
    });
    return;
  }

  const filePath = `${LOGS_DIR}/${filename}`;
  const fileMessages = await readLogFile(filePath);
  const memMessages = messagesMap[filename] || [];
  const allMessages = fileMessages.concat(memMessages);

  const matches = allMessages.filter(
    (m) => m.text && m.text.toLowerCase().includes(query.toLowerCase())
  );
  const recentMatches = matches.slice(-10);

  if (recentMatches.length === 0) {
    await client.sendMessage(entityId, {
      message: `No matches found for "${query}".`,
      replyTo: msg.id,
    });
    return;
  }

  const searchResults = recentMatches
    .map(
      (m) =>
        `[${m.senderUsername || m.senderFirstName || 'User'} at ${m.date
        }]: ${m.text}`
    )
    .join('\n');

  await client.sendMessage(entityId, {
    message: `**Search results for "${query}":**\n${searchResults}`,
    replyTo: msg.id,
  });
}

/**
 * Handles the "fsearch" command to do a fuzzy search in today's logs.
 */
async function handleFuzzySearchCommand(entityId, filename, query, msg) {
  if (!query) {
    await client.sendMessage(entityId, {
      message: 'Please provide a query to fuzzy search for, e.g. `!fsearch something`',
      replyTo: msg.id,
    });
    return;
  }

  const filePath = `${LOGS_DIR}/${filename}`;
  const fileMessages = await readLogFile(filePath);
  const memMessages = messagesMap[filename] || [];
  const allMessages = fileMessages.concat(memMessages);

  const scoredMessages = allMessages
    .map((m) => ({
      ...m,
      score: fuzzyScore(m.text || '', query),
    }))
    .filter((m) => m.score > 0);

  scoredMessages.sort((a, b) => b.score - a.score);

  const topMatches = scoredMessages.slice(0, 10);
  if (topMatches.length === 0) {
    await client.sendMessage(entityId, {
      message: `No fuzzy matches found for "${query}".`,
      replyTo: msg.id,
    });
    return;
  }

  const results = topMatches
    .map(
      (m) =>
        `[${m.senderUsername || m.senderFirstName || 'User'} at ${m.date
        }]: ${m.text}`
    )
    .join('\n');

  await client.sendMessage(entityId, {
    message: `**Fuzzy search results for "${query}":**\n${results}`,
    replyTo: msg.id,
  });
}

/**
 * Handles the "/z" command, which spawns an external process (zindex4) to do compressed-file searching.
 */
async function handleZCommand(entityId, msg) {
  if (UNAUTHORIZED_IDS.includes(String(msg.senderId?.value))) {
    await client.sendMessage(entityId, {
      message: 'Nah.',
      replyTo: msg.id,
    });
    return;
  }
  const searchTerm = msg.text.slice('/z'.length).trim();
  if (!searchTerm) {
    await client.sendMessage(entityId, {
      message: 'Usage: /z <search-term>',
      replyTo: msg.id,
    });
    return;
  }

  const MAX_UNIQUE_LINES = 500; // adjust as desired

  let child;
  // if (disallowedRegex.test(searchTerm)) {
  //   child = spawn('./zindex5', [searchTerm]);
  // } else {
  child = spawn('./zindex5', [searchTerm]);
  // }

  let leftover = '';
  const uniqueLines = new Set();
  let doneReading = false;

  child.stdout.on('data', (chunk) => {
    if (doneReading) return;

    const dataStr = leftover + chunk.toString();
    const lines = dataStr.split('\n');
    leftover = lines.pop() || '';

    for (const line of lines) {
      // Everything after the substring "preview"
      const marker = '"preview"';
      let finalText = line.trim();
      const idx = finalText.indexOf(marker);
      if (idx === -1) continue;

      // Slice out everything before/including "preview"
      finalText = finalText.slice(idx + marker.length);

      //Remove some known suffixes
      finalText = finalText.replace('\n"}\n', '');
      finalText = finalText.replaceAll(',""', '')
      finalText = finalText.replace('"}', '');

      // Remove leading :"
      if (finalText.startsWith(':"')) {
        finalText = finalText.slice(2);
      }

      // Check for at least 2 quotes
      const quoteCount = (finalText.match(/:/g) || []).length;
      if (quoteCount < 2 && finalText.includes('https://')) {
        continue;
      }

      // Check that there are no runs of 3+ consecutive whitespaces
      if (/\s{3,}/.test(finalText)) {
        continue;
      }

      // Exclude lines containing URL: or TITLE:
      if (finalText.includes('URL:') || finalText.includes('TITLE:') || finalText.includes('\t')) {
        continue;
      }

      if (finalText.startsWith("{") && finalText.slice(1).startsWith("\"")) {
        try {
          const finalObj = JSON.parse(finalText)
          finalText = `${finalObj.first_name ?? '[--]'}:${finalObj.last_name ?? '[--]'}:${finalObj.username ?? '[--]'}:(${finalObj.email_domain ?? ''}):${finalObj.email}:${finalObj.password_plaintext ?? (finalObj.password ?? '[--]')}:https://${finalObj.target_domain ?? '[--]'}`
          uniqueLines.add(finalText);
        } catch (e) {
          uniqueLines.add(finalText);
        }

      } else {

        // Passed all checks, add to the set
        uniqueLines.add(finalText);
      }

      // Stop if we hit the max
      if (uniqueLines.size >= MAX_UNIQUE_LINES) {
        doneReading = true;
        child.kill('SIGTERM');
        break;
      }
    }

  });

  child.stdout.on('end', async () => {
    if (!doneReading && leftover.trim()) {
      uniqueLines.add(leftover.trim());
    }
    if (uniqueLines.size === 0) {
      await client.sendMessage(entityId, {
        message: `No results for "${searchTerm}".`,
        replyTo: msg.id,
      });
      return;
    }

    // Convert to array and join
    const finalLines = [...uniqueLines];
    const finalOutput = finalLines.join('\n');

    const resultsFilename = 'search_results.txt';
    await fs.writeFile(resultsFilename, finalOutput, 'utf-8');
    const fileData = Buffer.from(finalOutput, 'utf-8');

    await client.sendMessage(entityId, {
      message: `${finalLines.length} Search results for "${searchTerm}" (showing up to ${MAX_UNIQUE_LINES} unique lines).`,
      file: new CustomFile(resultsFilename, fileData.length, '', fileData),
      replyTo: msg.id,
    });
  });

  child.stderr.on('data', (chunk) => {
    console.error('[zindex4 stderr]', chunk.toString());
  });

  child.on('error', async (err) => {
    console.error('zindex4 spawn error:', err);
    await client.sendMessage(entityId, {
      message: `Error running search: ${err.message}`,
      replyTo: msg.id,
    });
  });

  child.on('close', (code) => {
    console.log(`zindex4 process exited with code ${code}`);
  });
}

/**
 * Barks. Yep.
 */
async function handleArfCommand(entityId, msg) {
  // If you want to restrict ARF usage, check user auth first:
  // if (!isUserAuthorized(msg.senderId?.value)) {...}

  try {
    for (let i = 0; i < 3; i++) {
      await client.sendMessage(entityId, {
        message: 'ARF!',
        replyTo: msg.id,
      });
      await sleep(1000);
    }
    console.log('ARF command executed successfully');
  } catch (error) {
    console.error('Error sending ARF messages:', error);
  }
}

/**
 * Handles "!w" command, fetches content from an endpoint and possibly sends PDF/inline.
 */
async function handleWCommand(entityId, parsedCommand, msg) {
  try {
    const coptions = parsedCommand.content.split(' ');
    let content = '';
    const options = { pdf: false, inline: false };

    // Very naive parse: first "chunk" is flags, second "chunk" is URL
    if (coptions.length > 1) {
      const flags = coptions[0].toLowerCase();
      options.pdf = flags.includes('+pdf');
      options.inline = flags.includes('+inline');
      content = coptions[1];
    } else {
      content = coptions[0];
    }

    const response = await sendPromptAndGetResponse(parsedCommand.command, content);

    // Check if the response is the { name, data, text } object (wizard)
    if (typeof response === 'object' && response.data && response.text) {
      if (options.pdf && options.inline) {
        const pdfBuffer = await convertMarkdownToPdf(response.text);
        await client.sendMessage(entityId, {
          message: `Content for ${content}:\n\n${response.text}`,
          file: new CustomFile(response.name + '.pdf', pdfBuffer.length, '', pdfBuffer),
          replyTo: msg.id,
        });
      } else if (options.pdf) {
        const pdfBuffer = await convertMarkdownToPdf(response.text);
        await client.sendMessage(entityId, {
          message: '```Markdown and PDF for ->> ' + content + '```',
          file: [
            new CustomFile(response.name + '.md', response.data.length, '', response.data),
            new CustomFile(response.name + '.pdf', pdfBuffer.length, '', pdfBuffer),
          ],
          replyTo: msg.id,
        });
      } else if (options.inline) {
        await client.sendMessage(entityId, {
          message: `Content for ${content}:\n\n${response.text.slice(0, 2000)}`,
          replyTo: msg.id,
        });
      } else {
        // Default behavior: send markdown file
        await client.sendMessage(entityId, {
          message: '```Markdown for ->> ' + content + '```',
          file: [
            new CustomFile(response.name + '.md', response.data.length, '', response.data),
          ],
          replyTo: msg.id,
        });
      }
    } else if (typeof response === 'string') {
      // If for some reason the endpoint returns just a string:
      await client.sendMessage(entityId, {
        message: response,
        replyTo: msg.id,
      });
    }
  } catch (error) {
    console.error('Error in !w command:', error);
  }
}

/**
 * Handles "!q" or "!b" commands, sending a prompt to an LLM endpoint.
 */
async function handleQAndBCommand(entityId, parsedCommand, msg) {
  const senderId = msg.senderId?.value;
  const senderName =
    msg.sender?.firstName || msg.sender?.lastName || msg.sender?.username || 'Unknown User';

  // Example auth check if desired:
  // if (!isUserAuthorized(senderId)) { ... }

  try {
    const llmcontext = await fs.readFile(LLM_CONTEXT_PATH, 'utf8');
    const useContext = parsedCommand.command === 'b' ? llmcontext : '';

    const response = await sendPromptAndGetResponse(
      parsedCommand.command,
      parsedCommand.content,
      useContext
    );

    console.log('Automated reply:', response);
    await client.sendMessage(entityId, {
      message: `${COMMANDS[parsedCommand.command].prefix} ${response}`,
      replyTo: msg.id,
    });
    console.log('Automated reply sent successfully');
  } catch (error) {
    console.error('Error sending Q/B automated reply:', error);
  }
}

/* ------------------------------------------------------------------
   Main Event Handler
   ------------------------------------------------------------------ */

/**
 * Main new-message event handler. Decides how to respond or log a message.
 * @param {NewMessage.Event} event
 */
async function handleNewMessage(event) {
  const msg = event.message;
  const text = msg.message || '';

  // Detect the type of message (channel, user, group)
  let messageType = 'unknown';
  let entityId;
  if (msg?.peerId?.channelId) {
    entityId = Number(msg.peerId.channelId.value);
    messageType = 'channel';
  } else if (msg?.peerId?.userId) {
    entityId = Number(msg.peerId.userId.value);
    messageType = msg.out ? 'outgoing' : 'direct';
  } else if (msg?.peerId?.chatId) {
    entityId = Number(msg.peerId.chatId.value);
    messageType = 'group';
  }

  console.log(`Received a ${messageType} message:`, text);
  if (!entityId) {
    console.log('Unable to determine the entity ID for this message');
    return;
  }

  // Resolve entity name or fallback to ID
  let entityNameOrId = entityCache[entityId];
  if (!entityNameOrId) {
    try {
      // Attempt to get entity. Channels often require string IDs, while users/chats can be numeric.
      let entity;
      if (messageType === 'channel') {
        entity = await client.getEntity(entityId.toString());
      } else {
        entity = await client.getEntity(entityId);
      }

      if ('title' in entity) {
        entityNameOrId = entity.title;
      } else if ('username' in entity) {
        entityNameOrId = entity.username;
      } else if ('firstName' in entity) {
        entityNameOrId = entity.firstName;
      } else {
        entityNameOrId = String(entityId);
      }
      entityCache[entityId] = entityNameOrId;
    } catch (error) {
      console.error('Error fetching entity:', error);

      try {
        // If direct fetch fails, fetch all dialogs to update internal caches and try again
        await client.getDialogs();
        let entity;
        if (messageType === 'channel') {
          entity = await client.getEntity(entityId.toString());
        } else {
          entity = await client.getEntity(entityId);
        }

        if ('title' in entity) {
          entityNameOrId = entity.title;
        } else if ('username' in entity) {
          entityNameOrId = entity.username;
        } else if ('firstName' in entity) {
          entityNameOrId = entity.firstName;
        } else {
          entityNameOrId = String(entityId);
        }
        entityCache[entityId] = entityNameOrId;
      } catch (err2) {
        console.error('Error fetching entity after getDialogs:', err2);
        entityNameOrId = String(entityId);
        entityCache[entityId] = entityNameOrId;
      }
    }
  }

  // Log the message (in-memory and eventually to file)
  const date = new Date().toISOString().split('T')[0];
  const filename = `${entityNameOrId}-${date}.json`;

  // Convert or guess the date from Telegram (which may be a Date, number, or bigint).
  let dateString = '';
  if (msg.date instanceof Date) {
    dateString = msg.date.toISOString();
  } else if (typeof msg.date === 'number') {
    dateString = new Date(msg.date * 1000).toISOString();
  } else if (typeof msg.date === 'bigint') {
    dateString = new Date(Number(msg.date) * 1000).toISOString();
  } else {
    dateString = new Date().toISOString();
  }

  const messageData = {
    messageId: msg.id,
    date: dateString,
    text: msg.message,
    senderId: msg.senderId?.value,
    senderUsername: msg.sender?.username,
    senderFirstName: msg.sender?.firstName,
    senderLastName: msg.sender?.lastName,
  };

  messagesMap[filename] = messagesMap[filename] || [];
  messagesMap[filename].push(messageData);

  // Command detection
  let parsedCommand = parseCommand(text);

  // Support "reply-chaining": if I reply to my own message that had a command
  if (!parsedCommand && msg.replyTo?.replyToMsgId) {
    try {
      const [repliedMessage] = await client.getMessages(entityId, {
        ids: [msg.replyTo.replyToMsgId],
        limit: 1,
      });
      if (repliedMessage.sender?.username === MY_USERNAME) {
        // The original message from me might contain a command prefix
        const fallbackCommand = parseCommand(repliedMessage.text || '');
        if (fallbackCommand) {
          fallbackCommand.content = text;
          parsedCommand = fallbackCommand;
        }
      }
    } catch (error) {
      console.error('Error fetching replied message:', error);
    }
  }

  /* -------------------- Dispatch to Command Handlers -------------------- */

  if (!parsedCommand) return; // No command recognized

  switch (parsedCommand.command) {
    // Summarize
    case 'summarize':
      await handleSummarizeCommand(entityId, entityNameOrId, msg);
      break;

    // Stats
    case 'stats':
      await handleStatsCommand(entityId, filename, msg);
      break;

    // Search
    case 'search':
      await handleSearchCommand(
        entityId,
        filename,
        parsedCommand.content.trim(),
        msg
      );
      break;

    // Fuzzy Search
    case 'fsearch':
      await handleFuzzySearchCommand(
        entityId,
        filename,
        parsedCommand.content.trim(),
        msg
      );
      break;

    // Z Command
    case 'z':
      await handleZCommand(entityId, msg);
      break;

    // W command
    case 'w':
      await handleWCommand(entityId, parsedCommand, msg);
      break;

    // ARF command
    case 'arf':
      await handleArfCommand(entityId, msg);
      break;

    // Q or B commands
    case 'q':
    case 'b':
      await handleQAndBCommand(entityId, parsedCommand, msg);
      break;

    default:
      // Unhandled command
      break;
  }
}

/* ------------------------------------------------------------------
   Main Application
   ------------------------------------------------------------------ */

(async () => {
  console.log('Starting...');
  if (await login()) {
    console.log('Telegram session established.');
    console.log('Session string:', client.session.save());

    // Add event handler for incoming messages
    client.addEventHandler(
      handleNewMessage,
      new NewMessage({ incoming: true, forwards: false })
    );

    // Save unsaved messages to file every X minutes
    setInterval(saveMessagesToFile, SAVE_INTERVAL_MS);

    // Keep process alive
    await new Promise(() => { });
  } else {
    console.log('Failed to log in. Exiting...');
  }
})();
