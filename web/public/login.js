// Login form handler - CSP uyumlu
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('loginForm');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');
    const loginBtn = document.getElementById('loginBtn');
    const errorMessage = document.getElementById('errorMessage');

    // Form submit event listener
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        await handleLogin();
    });

    // Enter tuşu ile form gönderimi
    passwordInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            handleLogin();
        }
    });

    // Login fonksiyonu
    async function handleLogin() {
        const username = usernameInput.value.trim();
        const password = passwordInput.value;

        // Client-side validation
        if (!username || username.length < 3) {
            showError('Kullanıcı adı en az 3 karakter olmalıdır');
            return;
        }

        if (!password || password.length < 6) {
            showError('Şifre en az 6 karakter olmalıdır');
            return;
        }

        // Butonu devre dışı bırak ve loading göster
        loginBtn.disabled = true;
        loginBtn.innerHTML = '<span class="loading"></span> Giriş yapılıyor...';
        hideError();

        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    username: username,
                    password: password
                }),
                credentials: 'same-origin' // Cookie için gerekli
            });

            const data = await response.json();

            if (response.ok && data.ok) {
                // Başarılı login
                loginBtn.innerHTML = '✅ Başarılı! Yönlendiriliyor...';
                loginBtn.style.background = 'linear-gradient(135deg, #4CAF50 0%, #45a049 100%)';

                // Kısa bir gecikme sonra yönlendir
                setTimeout(() => {
                    window.location.href = '/';
                }, 500);
            } else {
                // Hatalı giriş
                showError(data.error || 'Geçersiz kullanıcı adı veya şifre');
                resetButton();
            }

        } catch (error) {
            console.error('Login error:', error);
            showError('Bağlantı hatası. Lütfen tekrar deneyin.');
            resetButton();
        }
    }

    // Hata mesajını göster
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.add('show');
    }

    // Hata mesajını gizle
    function hideError() {
        errorMessage.classList.remove('show');
    }

    // Butonu sıfırla
    function resetButton() {
        loginBtn.disabled = false;
        loginBtn.innerHTML = 'Giriş Yap';
        loginBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
    }

    // Input focus animasyonları
    [usernameInput, passwordInput].forEach(input => {
        input.addEventListener('focus', function() {
            this.parentElement.style.transform = 'scale(1.02)';
        });

        input.addEventListener('blur', function() {
            this.parentElement.style.transform = 'scale(1)';
        });
    });
});