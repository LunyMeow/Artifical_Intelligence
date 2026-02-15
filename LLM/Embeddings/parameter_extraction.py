#!/usr/bin/env python3
"""
Comprehensive parameter extraction and template substitution.
Maps user input to command templates and extracts parameters.
"""

import re
from typing import Dict, List, Tuple, Optional

class ParameterExtractor:
    """Extract parameters from user sentences and match to command templates."""
    
    # Common parameter patterns
    PATTERNS = {
        '<FILE>': [
            r'(?:file|dosya|document|doc|config)\s+([a-zA-Z0-9._-]+)',
            r'([a-zA-Z0-9._-]+\.(?:txt|py|csv|json|conf|log))',
            r'([\.][a-zA-Z0-9._-]+)',  # Hidden files
        ],
        '<DIR>': [
            r'(?:directory|folder|klasor|dizin)\s+([a-zA-Z0-9._/-]+)',
            r'(?:to|in|den)\s+([a-zA-Z0-9._/-~]+)(?:\s|$)',
            r'~|\.|\.\.|/|\w+/',
        ],
        '<SRC>': [
            r'(?:copy|move|from|den|from|kaynak)\s+([a-zA-Z0-9._/-]+)',
            r'([a-zA-Z0-9._/-]+)\s+(?:to|towards|e|ile)',
        ],
        '<DST>': [
            r'(?:to|into|destination|hedefe)\s+([a-zA-Z0-9._/-]+)',
            r'(?:as|olarak)\s+([a-zA-Z0-9._/-]+)',
        ],
        '<PATTERN>': [
            r'(?:search|find|ara)\s+(?:for\s+)?([a-zA-Z0-9.*+?^${}()|[\]\\-]+)',
            r'"([^"]+)"',  # Quoted pattern
        ],
        '<PERMS>': [
            r'(?:permissions|perms|izin)\s+([0-7]{3})',
            r'([0-7]{3})(?:\s|$)',
        ],
        '<OWNER>': [
            r'(?:owner|sahip)\s+([a-zA-Z0-9_-]+)',
            r'([a-zA-Z0-9_-]+):([a-zA-Z0-9_-]+)',  # user:group
        ],
    }
    
    COMMAND_KEYWORDS = {
        'cp': ['copy', 'duplicate', 'kopyala'],
        'mv': ['move', 'rename', 'taşı', 'yeniden adlandır'],
        'rm': ['delete', 'remove', 'remove sil', 'kaldır'],
        'mkdir': ['create', 'make', 'new', 'folder', 'directory', 'oluştur'],
        'cd': ['change', 'goto', 'go', 'directory', 'dizin', 'git'],
        'ls': ['list', 'show', 'display', 'ls', 'dir', 'listele', 'göster'],
        'cat': ['show', 'display', 'read', 'print', 'view', 'dosyayı'],
        'grep': ['search', 'find', 'grep', 'ara'],
        'chmod': ['permission', 'perms', 'change', 'izin'],
        'chown': ['owner', 'own', 'sahip'],
        'tar': ['archive', 'tar', 'compress'],
        'zip': ['zip', 'compress', 'sıkıştır'],
        'nano': ['edit', 'nano', 'vi', 'düzenle'],
        'vi': ['edit', 'vi', 'vim', 'düzenle'],
        'pwd': ['path', 'present', 'working', 'konum'],
        'bash': ['bash', 'shell', 'terminal', 'console', 'kabuğu'],
    }
    
    def __init__(self):
        self.last_extracted = {}
    
    def extract_command(self, sentence: str) -> Optional[str]:
        """Detect which command is being requested."""
        sentence_lower = sentence.lower()
        
        for cmd, keywords in self.COMMAND_KEYWORDS.items():
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    return cmd
        
        return None
    
    def extract_parameters(self, sentence: str, command: str) -> Dict[str, str]:
        """Extract parameters from sentence for given command."""
        params = {}
        sentence_lower = sentence.lower()
        
        # Command-specific extraction
        if command == 'cp' or command == 'copy':
            src = self._extract_parameter(sentence, '<SRC>')
            dst = self._extract_parameter(sentence, '<DST>')
            if src:
                params['<SRC>'] = src
            if dst:
                params['<DST>'] = dst
        
        elif command == 'mv' or command == 'move':
            src = self._extract_parameter(sentence, '<SRC>')
            dst = self._extract_parameter(sentence, '<DST>')
            if src:
                params['<SRC>'] = src
            if dst:
                params['<DST>'] = dst
        
        elif command == 'rm' or command == 'delete':
            file = self._extract_parameter(sentence, '<FILE>')
            if file:
                params['<FILE>'] = file
        
        elif command == 'mkdir':
            dir = self._extract_parameter(sentence, '<DIR>')
            if dir:
                params['<DIR>'] = dir
        
        elif command == 'cd':
            dir = self._extract_parameter(sentence, '<DIR>')
            if dir:
                params['<DIR>'] = dir
        
        elif command == 'ls':
            dir = self._extract_parameter(sentence, '<DIR>')
            if dir:
                params['<DIR>'] = dir
            else:
                params['<DIR>'] = '.'
        
        elif command == 'cat':
            file = self._extract_parameter(sentence, '<FILE>')
            if file:
                params['<FILE>'] = file
        
        elif command == 'grep':
            pattern = self._extract_parameter(sentence, '<PATTERN>')
            file = self._extract_parameter(sentence, '<FILE>')
            if pattern:
                params['<PATTERN>'] = pattern
            if file:
                params['<FILE>'] = file
        
        elif command in ['nano', 'vi', 'vim']:
            file = self._extract_parameter(sentence, '<FILE>')
            if file:
                params['<FILE>'] = file
        
        self.last_extracted = params
        return params
    
    def _extract_parameter(self, sentence: str, param_type: str) -> Optional[str]:
        """Extract single parameter using regex patterns."""
        if param_type not in self.PATTERNS:
            return None
        
        patterns = self.PATTERNS[param_type]
        
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                return match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
        
        return None
    
    def fill_template(self, template: str, params: Dict[str, str]) -> str:
        """Replace template tags with extracted parameters."""
        result = template
        for tag, value in params.items():
            result = result.replace(tag, value)
        return result
    
    def process(self, sentence: str) -> Tuple[str, Dict[str, str]]:
        """Full pipeline: detect command, extract parameters, return filled template."""
        command = self.extract_command(sentence)
        if not command:
            return "", {}
        
        params = self.extract_parameters(sentence, command)
        
        # Get template for command
        templates = {
            'cp': 'cp <SRC> <DST>',
            'move': 'mv <SRC> <DST>',
            'mv': 'mv <SRC> <DST>',
            'rm': 'rm <FILE>',
            'delete': 'rm <FILE>',
            'mkdir': 'mkdir <DIR>',
            'cd': 'cd <DIR>',
            'ls': 'ls <DIR>',
            'cat': 'cat <FILE>',
            'grep': 'grep <PATTERN> <FILE>',
            'chmod': 'chmod <PERMS> <FILE>',
            'chown': 'chown <OWNER> <FILE>',
            'nano': 'nano <FILE>',
            'vi': 'vi <FILE>',
            'pwd': 'pwd',
            'bash': 'bash',
        }
        
        template = templates.get(command, '')
        if not template:
            return command, params
        
        filled = self.fill_template(template, params)
        return filled + " <end>", params


def test_extractor():
    """Test parameter extraction."""
    extractor = ParameterExtractor()
    
    test_cases = [
        "copy backup to projects",
        "backup dosyasını projelere kopyala",
        "copy file.txt to backup folder",
        "move important.doc to archive",
        "delete temp file",
        "create new folder",
        "change to home directory",
        "list files in current directory",
        "show config file",
        "search for error in log",
    ]
    
    print("=" * 70)
    print(" 🧪 PARAMETER EXTRACTION TESTS")
    print("=" * 70)
    
    for sentence in test_cases:
        cmd = extractor.extract_command(sentence)
        params = extractor.extract_parameters(sentence, cmd) if cmd else {}
        result, filled_params = extractor.process(sentence)
        
        print(f"\n📝 Input: {sentence}")
        print(f"   Command: {cmd}")
        print(f"   Params: {filled_params}")
        print(f"   Output: {result}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_extractor()
