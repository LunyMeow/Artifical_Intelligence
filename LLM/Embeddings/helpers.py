import re

class helpers:
    def normalize_command_params(tokens):
        """
        Komut parametrelerini tip placeholderlarına dönüştürür.
        createEmbeddings.py'deki fonksiyonla TAMAMEN AYNI olmalıdır.
        """
        normalized = []
        
        # Known commands (keep as-is)
        common_commands = [
            "mkdir", "rm", "cd", "ls", "cp", "mv", "touch", "cat",
            "grep", "find", "chmod", "chown", "sudo", "apt", "git",
            "python", "python3", "node", "npm", "bash", "sh", "echo",
            "tar", "zip", "unzip", "wget", "curl", "ssh", "scp"
        ]
        
        for token in tokens:
            # Control tokens (keep as-is)
            if token in ["<end>"]:
                normalized.append(token)
                continue
            
            if token in common_commands:
                normalized.append(token)
                continue
            
            # Flags (keep as-is)
            if token.startswith("-"):
                normalized.append(token)
                continue
            
            # Parameter type detection
            # 1. Paths (absolute or relative with slashes)
            if re.match(r"^[/\\]|.*[/\\].*", token):
                normalized.append("<PATH>")
            # 2. Files with extensions
            elif re.match(r".*\.[a-zA-Z0-9]+$", token):
                normalized.append("<FILE>")
            # 3. Pure numbers
            elif re.match(r"^\d+$", token):
                normalized.append("<NUMBER>")
            # 4. IP addresses
            elif re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", token):
                normalized.append("<IP>")
            # 5. URLs
            elif re.match(r"^https?://", token):
                normalized.append("<URL>")
            # 6. Environment variables
            elif re.match(r"^\$[A-Z_]+$", token):
                normalized.append("<VAR>")
            # 7. Generic directory/file names (no extension, no path)
            else:
                normalized.append("<DIR>")
        
        return normalized