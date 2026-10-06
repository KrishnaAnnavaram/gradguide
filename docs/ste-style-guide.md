# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of gradguide and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, "test" is a noun or a verb, "check" is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: "prepare", "do", "find", "get", "make".
4. Do not use an "-ing" form as a noun or an adjective ("the running job", "after indexing").
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, "gotcha", "bells and whistles").
6. Do not use `and/or`. Write "A, B or both".
7. Do not use `should`, `could`, `would` or `may` for instructions. Use "must" for a rule, the
   imperative for a step and "can" for a possibility.
8. Keep the articles "a", "an" and "the" in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: "Run the tests." Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: "If the index is stale, build it again."
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase ("The cost model") or an imperative ("Run the demo").
   Do not start a heading with an "-ing" form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or "check that" |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the gradguide documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **student** | The person who asks a question: a graduate student. | user (for the person), reader, customer |
| **institution** | The school whose documents gradguide reads. gradguide names no real institution. | university (as a name), college |
| **document** | One source file in the document folder: `.md`, `.markdown`, `.txt`, `.json`, `.jsonl`, `.html`, `.htm` or `.pdf`. | file (for a source), article, page |
| **document folder** | The folder that gradguide reads documents from (`GRADGUIDE_DOCS_DIR`, default `sample_docs`). | corpus folder, knowledge base |
| **corpus** | All documents in the document folder, as one set. | knowledge base, dataset |
| **source manifest** | The optional file `sources.toml` in the document folder. It gives the owner, the refresh period and the update date of each document. | sources file, registry |
| **front-matter** | The `---` block at the top of a Markdown or text document, for example with `title` and `updated`. | header, YAML |
| **section** | A loader output: the text under one heading, or one FAQ pair. | part, block |
| **FAQ pair** | One question and its answer from a FAQ JSON or JSONL document. It is a section of kind `faq`. | Q/A record, entry |
| **heading path** | The headings above a section, joined with ` > `, for example `Registration and Enrollment > Registration Holds`. | breadcrumb, title |
| **chunk** | One unit of text that gradguide stores and retrieves. A chunk is part of one section. | segment, piece, split, node |
| **chunk ID** | The first 16 hex characters of the SHA-256 of the source, the heading path and the chunk text. | doc id, index number |
| **token** | The unit that the token counter counts to measure the chunk size and the history budget. | word (for this unit), character |
| **token counter** | The `simple` counter (words and punctuation marks) or the `tiktoken` counter (`cl100k_base`). | tokenizer (in prose), splitter |
| **search term** | A lower-case word, without stopwords and with plurals folded, that BM25, the hashing embedder and the lexical reranker use. | token, keyword |
| **overlap** | The last words of a chunk that the next chunk of the same section starts with. | carry-over, stride |
| **index** | The four files in the index folder: `chunks.jsonl`, `bm25.json`, `vectors.npy` and `manifest.json`. | database, vector store, cache |
| **index manifest** | The file `manifest.json` in the index. It holds the fingerprint, the settings and the hash of each document. | metadata file |
| **fingerprint** | A SHA-256 value of the index settings and of the hash of each document and of the source manifest. | checksum, signature |
| **sync** | One run of `sync_index`: load the index, or embed new chunks, reuse vectors and remove old chunks. | rebuild (when only part changes), refresh |
| **embedder** | The provider that changes text into a vector. | encoder, embedding service |
| **vector** | The list of numbers that the embedder gives for one text. Each vector has a length of 1. | embedding (as a noun), dense representation |
| **chat model** | The provider that writes the answer from the prompt. | LLM (in prose), generator, bot |
| **provider** | An embedder or a chat model behind one interface: offline, `openai` or `ollama`. | backend, engine, vendor |
| **offline provider** | The hashing embedder or the echo chat model. Both work with no key and no network. | fake, mock, stub (in prose) |
| **mode** | One retrieval path: `bm25`, `vector`, `hybrid` or `hybrid+rerank`. | strategy, method |
| **candidate** | A chunk that BM25 or vector search returns before fusion. | result, match |
| **fusion** | Reciprocal-rank fusion (RRF): one ranked list from the BM25 list and the vector list. | merge, blend |
| **reranker** | The optional second stage that scores the fused candidates again: `lexical` or `cross-encoder`. | re-ranker, ranker, scorer |
| **hit** | A chunk that the retriever returns, with its scores and its ranks. | result, document |
| **passage** | A hit that gradguide puts in the prompt in a `<passage id="n">` block. | context chunk, snippet, excerpt |
| **evidence score** | The higher value of the search-term coverage and the cosine score of a hit. | relevance (alone), confidence |
| **abstention** | The fixed answer that sends the student to the fallback contact when no hit has a sufficient evidence score. | refusal, rejection, fallback answer |
| **fallback contact** | The office that the abstention names (`GRADGUIDE_FALLBACK_CONTACT`). | help desk, support |
| **prompt template** | A versioned text file `generate/templates/<kind>_<version>.txt`. | prompt file |
| **prompt version** | The version part of the prompt template name, for example `v1`. Each answer records it. | template id |
| **marker** | A citation number in square brackets in the answer, for example `[2]`. | reference, footnote |
| **citation** | A valid marker together with the chunk that it points to. | source (alone), reference |
| **source list** | The `Sources:` block after the answer. It has one line for each citation. | bibliography, references |
| **history budget** | The maximum number of tokens of earlier exchanges in the prompt (`GRADGUIDE_HISTORY_TOKENS`). | memory, context window |
| **query log** | The optional file `query_log.jsonl` with redacted questions. | audit log, history |
| **rate limiter** | The per-client limit of `/ask` requests in one minute. | throttle, quota |
| **gold set** | The file `eval/gold_set.jsonl`: answerable questions with targets, and unanswerable questions. | test set, benchmark, ground truth |
| **target** | One expected `source` and `heading` pair for a gold-set question. | label, answer key |
| **ablation** | One evaluation run for each value of a setting, with all other settings unchanged. | sweep, experiment |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **load** | Read the documents from the document folder, or read a saved index from the index folder. |
| **split** | Cut a document into sections, or cut a section into chunks. |
| **count** | Get the number of tokens of a text with the token counter. |
| **embed** | Change text into a vector with the embedder. |
| **sync** | Make the index match the corpus: embed new chunks, reuse vectors, remove old chunks. |
| **retrieve** | Get the hits for a question from the index. |
| **fuse** | Make one ranked list from two ranked lists with reciprocal-rank fusion. |
| **rerank** | Score the fused candidates again with the reranker and sort them by that score. |
| **abstain** | Give the abstention and do not send a prompt to the chat model. |
| **neutralize** | Change `<` and `>` of `passage`, `passages` and `question` tags in untrusted text to `(` and `)`. |
| **cite** | Put a marker in the answer for a passage. |
| **validate** | Check each marker against the passages and remove a marker that points to no passage. |
| **escape** | Replace the HTML characters `<`, `>`, `&` and quotes with HTML entities. |
| **redact** | Replace e-mail addresses, phone numbers and numbers of 5 or more digits with `[email]`, `[phone]` and `[number]`. |
| **purge** | Remove query-log records that are older than the retention period. |
| **evaluate** | Measure the retrieval and the abstention decisions on the gold set. |
