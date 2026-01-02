import express from "express";
import cookieParser from "cookie-parser";
import jwt from "jsonwebtoken";
import path from "path";
import rateLimit from 'express-rate-limit';
import crypto from 'crypto'; // ✅ EKLE

const app = express();
const SECRET = process.env.JWT_SECRET || crypto.randomBytes(64).toString('hex');
if (!process.env.JWT_SECRET) {
    console.error('⚠️  JWT_SECRET tanımlı değil!');
}

app.use(cookieParser());
app.use(express.json());
app.use(express.urlencoded({ extended: true })); // ✅ EKLE - body parsing için

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100
});

app.use('/api/', limiter);

function auth(req, res, next) {
    try {
        jwt.verify(req.cookies.auth, SECRET);
        next();
    } catch {
        res.sendStatus(401);
    }
}

/* 🔓 PUBLIC WASM RUNTIME */
app.use("/wasm", express.static(path.join("public", "wasm")));

/* 🔒 MODEL DOSYALARI */
app.get("/model/:file", auth, (req, res) => {
    const allowed = [
        "command_model.bin",
        "command_model.meta"
    ];

    const safePath = path.basename(req.params.file);
    if (!allowed.includes(safePath))
        return res.sendStatus(403);

    res.sendFile(path.resolve("user_0000", safePath));
});

/* AUTH CHECK */
app.get("/api/me", auth, (req, res) => {
    res.json({ ok: true });
});

/* LOGIN */
app.post("/api/login", (req, res) => {
    const { username, password } = req.body;

    if (username === "admin" && password === "admin") {
        const token = jwt.sign({ username }, SECRET, { expiresIn: '24h' });
        res.cookie('auth', token, {
            httpOnly: true,
            secure: false, // ⚠️ Development için false, production'da true
            sameSite: 'lax', // ⚠️ 'strict' yerine 'lax' (local test için)
            maxAge: 86400000
        });
        res.json({ ok: true });
    } else {
        res.sendStatus(401);
    }
});

/* NONCE */
app.get("/api/nonce", auth, (req, res) => {
    const nonce = crypto.randomBytes(16).toString('hex');
    res.json({ nonce });
});

app.get("/api/a", auth, (req, res) => {
    res.sendFile(
        path.resolve("public", "a.html")
    );
});


/* STATIC */
app.use(express.static("public"));

app.listen(3000, () => console.log("✅ Server running on http://localhost:3000"));