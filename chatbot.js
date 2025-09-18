/*
 AI Edu Chatbot – Node.js server with frontend integration
*/

const path = require("path");
const express = require("express");
const bodyParser = require("body-parser");
const dotenv = require("dotenv");
const Groq = require("groq-sdk");
const multer = require("multer"); // 🆕 for file uploads

dotenv.config();

const app = express();
app.use(bodyParser.json());

// ✅ Serve static frontend from public/
app.use(express.static(path.join(__dirname, "public")));

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

// -----------------------------
// Chat endpoint (AI Tutor)
// -----------------------------
app.post("/chat", async (req, res) => {
  try {
    const { message, grade } = req.body;
    if (!message) return res.status(400).json({ error: "Message is required" });

    let gradeInstruction = "";

    if (grade && grade !== "general") {
      gradeInstruction = `The student is in Grade ${grade}. 
      ✅ Explain the answer at a Grade ${grade} level (use simple words for lower grades, 
      and more depth for higher grades). 
      ✅ After your explanation, suggest 2–3 related topics from their grade's curriculum.`;
    } else {
      gradeInstruction = `The user did not specify a grade. Provide a general educational answer suitable for all levels.`;
    }

    const response = await groq.chat.completions.create({
      model: "llama3-8b-8192",
      messages: [
        { role: "system", content: "You are a helpful educational tutor." },
        { role: "system", content: gradeInstruction },
        { role: "user", content: message },
      ],
    });

    const reply =
      response.choices?.[0]?.message?.content?.trim() ||
      "Sorry, I couldn't generate a response.";

    res.json({ reply });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal error", detail: error.message });
  }
});

// -----------------------------
// 🆕 Image Upload + Analysis
// -----------------------------
const storage = multer.memoryStorage(); 
const upload = multer({ storage: storage });

app.post("/upload-image", upload.single("photo"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ reply: "No image uploaded." });
    }

    const fileName = req.file.originalname;
    const fileSizeKB = (req.file.size / 1024).toFixed(2);

    // 🔮 Replace with real AI image analysis (Groq doesn’t support vision yet)
    // For now, simulate feedback
    const feedback = `✅ Image "${fileName}" received (size: ${fileSizeKB} KB).\n
    🔍 Analysis: Looks like an educational resource.\n
    💡 Feedback: You can use this image to enhance your learning material!`;

    res.json({ reply: feedback });
  } catch (error) {
    console.error(error);
    res.status(500).json({ reply: "Error analyzing image." });
  }
});

// -----------------------------
// Quiz endpoint (same as before)
// -----------------------------
// ... keep your /quiz code here ...

// -----------------------------
// Health check
// -----------------------------
app.get("/health", (req, res) => {
  res.json({ ok: true, service: "ai-edu-chatbot", time: new Date().toISOString() });
});

// ✅ Route `/` to `index.html`
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// -----------------------------
// Start server
// -----------------------------
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 AI Edu Chatbot running at http://localhost:${PORT}`);
});
