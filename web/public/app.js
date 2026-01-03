import createModule from "/wasm/model.js";

let Module, loadModel, runInference;
let currentUser = null; // 👈 Mevcut kullanıcı bilgisi

// Status güncelleme fonksiyonu
function updateStatus(message, type) {
    const statusEl = document.getElementById('status');
    statusEl.innerHTML = message;

    const statusContainer = statusEl.closest('.status-container');
    statusContainer.style.borderLeftColor =
        type === 'ready' ? '#4CAF50' :
        type === 'error' ? '#F44336' :
        '#667eea';
}

// Kimlik doğrulama kontrolü
async function checkAuth() {
    try {
        updateStatus('<span class="loading"></span> Kimlik doğrulanıyor...', 'loading');
        const r = await fetch("/api/me");
        if (!r.ok) {
            location.href = "/login.html";
            return false;
        }

        const userData = await r.json();
        currentUser = userData; // 👈 Kullanıcı bilgisini sakla

        updateStatus(`<span class="loading"></span> Hoş geldin ${userData.username}, model yükleniyor...`, 'loading');
        return true;
    } catch (e) {
        console.error("Auth check failed:", e);
        location.href = "/login.html";
        return false;
    }
}

// Typing effect fonksiyonu
function typeWriter(text, element, speed = 30) {
    return new Promise((resolve) => {
        let i = 0;
        element.innerHTML = '';

        const cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        element.appendChild(cursor);

        function type() {
            if (i < text.length) {
                element.removeChild(cursor);
                element.innerHTML += text.charAt(i);
                element.appendChild(cursor);
                i++;
                setTimeout(type, speed);
            } else {
                element.removeChild(cursor);
                resolve();
            }
        }

        type();
    });
}

// Güvenli inferans fonksiyonu (nonce ile)
async function secureInference(text) {
    try {
        const r = await fetch("/api/nonce");
        if (!r.ok) throw new Error("Nonce fetch failed");
        const { nonce } = await r.json();
        return runInference(text);
    } catch (e) {
        console.error("Secure inference error:", e);
        throw new Error("Güvenli inferans başarısız: " + e.message);
    }
}

// Inferans çalıştırma fonksiyonu
async function runInferenceHandler() {
    try {
        const input = document.getElementById('input');
        const outputEl = document.getElementById('out');
        const btn = document.getElementById('btn');

        if (!input.value.trim()) {
            outputEl.textContent = 'Lütfen geçerli bir girdi girin.';
            return;
        }

        const originalText = btn.textContent;
        btn.textContent = 'İşleniyor...';
        btn.disabled = true;

        // İşlem sürüyor mesajını göster
        outputEl.innerHTML = '<span class="loading" style="width: 16px; height: 16px;"></span> İşleniyor...';

        // Güvenli inferansı çalıştır
        const result = await secureInference(input.value);

        // Sonucu harf harf yaz
        await typeWriter(result, outputEl, 20);

        btn.textContent = originalText;
        btn.disabled = false;

    } catch (e) {
        console.error("INFERENCE ERROR:", e);
        const outputEl = document.getElementById('out');
        outputEl.innerHTML = '❌ Hata: ' + e.message;

        const btn = document.getElementById('btn');
        btn.textContent = 'Inferans Çalıştır';
        btn.disabled = false;
    }
}

// Ana başlatma fonksiyonu
async function init() {
    try {
        // Önce kimlik doğrulama
        const authenticated = await checkAuth();
        if (!authenticated) return;

        // 👇 Kullanıcının model klasörünü göster
        console.log(`📁 Model klasörü: ${currentUser.modelFolder}`);

        // WASM modülünü yükle
        updateStatus('<span class="loading"></span> WASM modülü yükleniyor...', 'loading');
        Module = await createModule({
            locateFile: (p) => `/wasm/${p}`,
            onAbort(reason) {
                console.error("WASM ABORT:", reason);
                updateStatus('❌ Hata: Sistem durdu', 'error');
            },
            printErr(text) {
                console.error("WASM STDERR:", text);
            }
        });

        // Dizin oluştur (varsa hata vermesin)
        try {
            Module.FS.mkdir(`/${currentUser.modelFolder}`);
        } catch (e) {
            // Directory already exists → ignore
        }

        // Model dosyalarını indir (KULLANICIYA ÖZEL)
        updateStatus(`<span class="loading"></span> ${currentUser.username} için model indiriliyor...`, 'loading');

        const binResp = await fetch('/model/command_model.bin');
        if (!binResp.ok) throw new Error("Model binary yüklenemedi");
        const bin = await binResp.arrayBuffer();

        Module.FS.writeFile(
            `/${currentUser.modelFolder}/command_model.bin`,
            new Uint8Array(bin)
        );

        const metaResp = await fetch('/model/command_model.meta');
        if (!metaResp.ok) throw new Error("Model metadata yüklenemedi");
        const meta = await metaResp.text();

        Module.FS.writeFile(
            `/${currentUser.modelFolder}/command_model.meta`,
            meta
        );

        // WASM fonksiyonlarını bağla
        updateStatus('<span class="loading"></span> Model yükleniyor...', 'loading');

        loadModel = Module.cwrap(
            'load_user_model',
            null, ['string', 'string']
        );

        runInference = Module.cwrap(
            'run_inference',
            'string', ['string']
        );

        // Modeli yükle (kullanıcıya özel path)
        loadModel(
            `/${currentUser.modelFolder}/command_model.bin`,
            `/${currentUser.modelFolder}/command_model.meta`
        );

        // Başarılı
        updateStatus(`✅ Sistem hazır (${currentUser.username})`, 'ready');
        document.getElementById('btn').disabled = false;
        document.getElementById('input').focus();

    } catch (err) {
        console.error("INIT FAILED:", err);
        updateStatus('❌ Hata: ' + err.message, 'error');
    }
}

// Event listener'ları ekle
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('input');
    const btn = document.getElementById('btn');

    // Enter tuşu ile gönderme
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !btn.disabled) {
            runInferenceHandler();
        }
    });

    // Buton tıklama
    btn.addEventListener('click', runInferenceHandler);

    // Input focus efektleri
    input.addEventListener('focus', function() {
        this.style.borderColor = '#764ba2';
    });

    input.addEventListener('blur', function() {
        this.style.borderColor = '#e0e0e0';
    });

    // Global hata yakalama
    window.addEventListener("unhandledrejection", e => {
        console.error("UNHANDLED PROMISE:", e.reason);
        updateStatus('❌ Beklenmeyen hata', 'error');
    });

    window.addEventListener("error", e => {
        console.error("GLOBAL ERROR:", e.error || e.message);
        updateStatus('❌ Sistem hatası', 'error');
    });

    const logoutBtn = document.getElementById("logoutBtn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", logout);
    }

});


async function logout() {
    try {
        const r = await fetch("/api/logout", {
            method: "GET",
            credentials: "include"
        });

        // 200, 204, 302 hepsini kabul et
        if (![200, 204, 302].includes(r.status)) {
            throw new Error("Logout failed: " + r.status);
        }

        // Belleği temizle
        Module = null;
        loadModel = null;
        runInference = null;
        currentUser = null;

        location.href = "/login.html";
    } catch (e) {
        console.error("LOGOUT ERROR:", e);
        alert("Çıkış başarısız (backend reddetti)");
    }
}


// Sayfa yüklendiğinde başlat
init();