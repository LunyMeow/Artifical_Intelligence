import express from "express";
import cookieParser from "cookie-parser";
import jwt from "jsonwebtoken";
import path from "path";
import rateLimit from 'express-rate-limit';
import crypto from 'crypto';
import helmet from 'helmet';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import dotenv from 'dotenv';

// 🔧 .env dosyasını yükle
dotenv.config();



const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();

app.set('trust proxy', 1);


// 🔐 GÜVENLI SECRET YÖNETIMI
const SECRET = process.env.JWT_SECRET;
if (!SECRET || SECRET.length < 32) {
    console.error('❌ FATAL: JWT_SECRET eksik veya çok kısa (min 32 karakter)');
    console.error('📁 .env dosyası konumu:', path.resolve('.env'));
    console.error('🔍 Mevcut JWT_SECRET:', SECRET ? `${SECRET.length} karakter` : 'undefined');
    process.exit(1);
}

// 🛡️ HELMET - Güvenlik başlıkları
app.use(helmet({
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'", "'wasm-unsafe-eval'"],
            scriptSrcAttr: ["'none'"], // inline event handlers'ı engelle
            styleSrc: ["'self'", "'unsafe-inline'"], // CSS için gerekli
            imgSrc: ["'self'", "data:"],
            connectSrc: ["'self'"],
            workerSrc: ["'self'", "blob:"],
            fontSrc: ["'self'"],
            objectSrc: ["'none'"],
            baseUri: ["'self'"],
            formAction: ["'self'"]
        }
    },
    hsts: {
        maxAge: 31536000,
        includeSubDomains: true,
        preload: true
    }
}));

// 🚫 X-Powered-By başlığını gizle
app.disable('x-powered-by');

// 📦 Body parsing limitleri
app.use(express.json({ limit: '10kb' }));
app.use(express.urlencoded({ extended: true, limit: '10kb' }));
app.use(cookieParser());

// 🔒 Rate Limiting - Katmanlı koruma
const strictLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 dakika
    max: 5, // Login için çok katı
    message: { error: "Çok fazla deneme. Lütfen 15 dakika sonra tekrar deneyin." },
    standardHeaders: true,
    legacyHeaders: false
});

const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000,
    max: 100,
    message: { error: "Rate limit aşıldı." }
});

const modelLimiter = rateLimit({
    windowMs: 1 * 60 * 1000, // 1 dakika
    max: 10, // Model indirme için limit
    message: { error: "Model indirme limiti aşıldı." }
});

// 🔐 SHA-256 ile şifreleme
function hashPassword(password, salt) {
    return crypto.pbkdf2Sync(password, salt, 100000, 64, 'sha256').toString('hex');
}

// 🔑 KULLANICI VERİTABANI - Şifreler hash'lenmiş
const SALT = process.env.PASSWORD_SALT || crypto.randomBytes(16).toString('hex');
const USERS_DB = {
    "admin": {
        passwordHash: hashPassword("admin1234", SALT), // ⚠️ Production'da değiştirin!
        modelFolder: "user_0000",
        role: "admin",
        createdAt: Date.now()
    },
    "user1": {
        passwordHash: hashPassword("pass1234", SALT),
        modelFolder: "user_0001",
        role: "user",
        createdAt: Date.now()
    },
    "user2": {
        passwordHash: hashPassword("pass1234", SALT),
        modelFolder: "user_0002",
        role: "user",
        createdAt: Date.now()
    }
};

// 🛡️ Input validasyon
function sanitizeInput(input) {
    if (typeof input !== 'string') return '';
    return input.trim().slice(0, 100); // Max 100 karakter
}

function isValidUsername(username) {
    return /^[a-zA-Z0-9_]{3,20}$/.test(username);
}

function isValidPassword(password) {
    return typeof password === 'string' && password.length >= 6 && password.length <= 100;
}

// 🔐 Auth middleware - Güvenli
function auth(req, res, next) {
    try {
        const token = req.cookies.auth;
        
        if (!token) {
            return res.status(401).json({ error: "Kimlik doğrulaması gerekli" });
        }

        const decoded = jwt.verify(token, SECRET, {
            algorithms: ['HS256'], // Sadece HS256 algoritması
            maxAge: '24h'
        });

        // Kullanıcının hala geçerli olduğunu kontrol et
        if (!USERS_DB[decoded.username]) {
            return res.status(401).json({ error: "Geçersiz kullanıcı" });
        }

        req.user = decoded;
        next();
    } catch (err) {
        if (err.name === 'TokenExpiredError') {
            return res.status(401).json({ error: "Oturum süresi doldu" });
        }
        return res.status(401).json({ error: "Geçersiz token" });
    }
}

// 🔒 Path traversal koruması
function isPathSafe(userPath, basePath) {
    const resolved = path.resolve(basePath);
    const requested = path.resolve(userPath);
    return requested.startsWith(resolved);
}

// 🔓 PUBLIC WASM RUNTIME
app.use("/wasm", express.static(path.join(__dirname, "public", "wasm"), {
    maxAge: '1d',
    etag: true,
    lastModified: true
}));

// 🔒 MODEL DOSYALARI - KULLANICI BAZLI + Güvenli
app.get("/model/:file", auth, modelLimiter, async (req, res) => {
    try {
        const allowed = ["command_model","bpe_tokenizer.json"];
        const safePath = path.basename(req.params.file); // Path traversal koruması

        if (!allowed.includes(safePath)) {
            return res.status(403).json({ error: "Yetkisiz dosya erişimi" });
        }

        const userFolder = req.user.modelFolder;
        if (!userFolder || !USERS_DB[req.user.username]) {
            return res.status(404).json({ error: "Model klasörü bulunamadı" });
        }

        const baseDir = path.resolve(__dirname, userFolder);
        const filePath = path.join(baseDir, safePath);

        // Çift güvenlik kontrolü
        if (!isPathSafe(filePath, baseDir)) {
            console.error(`⚠️ Path traversal denemesi: ${req.user.username} -> ${req.params.file}`);
            return res.status(403).json({ error: "Güvenlik ihlali tespit edildi" });
        }

        // Dosya varlığı kontrolü
        const fs = await import('fs/promises');
        try {
            await fs.access(filePath);
        } catch {
            console.error(`❌ Dosya bulunamadı: ${filePath}`);
            return res.status(404).json({ error: "Model dosyası bulunamadı" });
        }

        res.sendFile(filePath, {
            maxAge: '1h',
            lastModified: true,
            headers: {
                'Cache-Control': 'private, max-age=3600'
            }
        });

    } catch (err) {
        console.error("Model erişim hatası:", err);
        res.status(500).json({ error: "Sunucu hatası" });
    }
});

// ✅ AUTH CHECK - Kullanıcı bilgisi
app.get("/api/me", auth, (req, res) => {
    res.json({
        ok: true,
        username: req.user.username,
        modelFolder: req.user.modelFolder,
        role: USERS_DB[req.user.username]?.role
    });
});

// 🔐 LOGIN - Güvenli
app.post("/api/login", strictLimiter, async (req, res) => {
    try {
        const username = sanitizeInput(req.body.username);
        const password = req.body.password;

        // Input validasyonu
        if (!isValidUsername(username)) {
            return res.status(400).json({ error: "Geçersiz kullanıcı adı formatı" });
        }

        if (!isValidPassword(password)) {
            return res.status(400).json({ error: "Geçersiz şifre formatı" });
        }

        // Kullanıcıyı bul
        const user = USERS_DB[username];

        // Timing attack koruması için sabit süre
        const passwordHash = hashPassword(password, SALT);
        const isValid = user && crypto.timingSafeEqual(
            Buffer.from(user.passwordHash, 'hex'),
            Buffer.from(passwordHash, 'hex')
        );

        if (!isValid) {
            // Generic hata mesajı (username/password leak önleme)
            await new Promise(resolve => setTimeout(resolve, 1000)); // Brute force koruması
            return res.status(401).json({ error: "Geçersiz kimlik bilgileri" });
        }

        // JWT oluştur
        const token = jwt.sign(
            {
                username: user ? username : '',
                modelFolder: user ? user.modelFolder : '',
                role: user ? user.role : '',
                iat: Math.floor(Date.now() / 1000)
            },
            SECRET,
            {
                expiresIn: '24h',
                algorithm: 'HS256',
                issuer: 'secure-ml-system',
                audience: 'ml-client'
            }
        );

        // Güvenli cookie ayarları
        res.cookie('auth', token, {
            httpOnly: true,
            secure: process.env.NODE_ENV === 'production', // Production'da HTTPS zorunlu
            sameSite: 'strict', // CSRF koruması
            maxAge: 86400000,
            path: '/'
        });

        // Başarılı login logu
        console.log(`✅ Login başarılı: ${username} (${new Date().toISOString()})`);

        res.json({
            ok: true,
            username,
            role: user.role
        });

    } catch (err) {
        console.error("Login hatası:", err);
        res.status(500).json({ error: "Sunucu hatası" });
    }
});

// 🚪 LOGOUT - Güvenli
app.get("/api/logout", auth, apiLimiter,(req, res) => {
    console.log(`🚪 Logout: ${req.user.username}`);
    
    res.clearCookie("auth", {
        httpOnly: true,
        sameSite: "strict",
        secure: process.env.NODE_ENV === 'production',
        path: '/'
    });

    res.json({ ok: true, message: "Çıkış başarılı" });
});

// 🎲 NONCE - Güvenli
app.get("/api/nonce", auth, apiLimiter, (req, res) => {
    const nonce = crypto.randomBytes(32).toString('hex'); // 32 byte = 256 bit
    res.json({ nonce });
});

// 📁 STATIC - Güvenli
app.use(express.static(path.join(__dirname, "public"), {
    maxAge: '1h',
    etag: true,
    lastModified: true,
    dotfiles: 'deny', // .env gibi dosyaları engelle
    index: 'index.html'
}));

// 🚫 404 Handler
app.use((req, res) => {
    res.status(404).json({ error: "Endpoint bulunamadı" });
});

// ⚠️ Error Handler
app.use((err, req, res, next) => {
    console.error("Sunucu hatası:", err);
    
    // Detaylı hata bilgisi sadece development'ta
    const errorResponse = process.env.NODE_ENV === 'production'
        ? { error: "Bir hata oluştu" }
        : { error: err.message, stack: err.stack };
    
    res.status(500).json(errorResponse);
});

// 🚀 Server başlat (Render uyumlu)
const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log("✅ Güvenli sunucu başlatıldı");
    console.log(`🌐 Port: ${PORT}`);
    console.log(`🔐 Ortam: ${process.env.NODE_ENV || 'development'}`);
    console.log("\n📁 Kullanıcı ve Model Klasörleri:");
    Object.entries(USERS_DB).forEach(([username, data]) => {
        console.log(`   - ${username} → ${data.modelFolder}`);
    });
    console.log("\n⚠️  UYARI: Production'da şifreleri değiştirin!");
    console.log("⚠️  UYARI: JWT_SECRET environment variable olarak ayarlayın!");
});
