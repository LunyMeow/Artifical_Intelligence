/**
 * @file ParameterExtractorV2.cpp
 * @brief Implementation of enhanced parameter extraction
 */

#include "ParameterExtractorV2.h"
#include <cctype>

void ParameterExtractorV2::init_command_keywords()
{
    command_keywords_["cp"] = {"copy", "duplicate", "kopyala", "cp"};
    command_keywords_["mv"] = {"move", "rename", "taşı", "mv", "yeniden adlandır"};
    command_keywords_["rm"] = {"delete", "remove", "sil", "kaldır", "rm"};
    command_keywords_["mkdir"] = {"create", "make", "new", "folder", "directory", "oluştur", "mkdir"};
    command_keywords_["cd"] = {"change", "goto", "go", "directory", "dizin", "cd", "git"};
    command_keywords_["ls"] = {"list", "show", "display", "ls", "dir", "listele", "göster"};
    command_keywords_["cat"] = {"show", "display", "read", "print", "view", "dosyayı", "cat"};
    command_keywords_["grep"] = {"search", "find", "grep", "ara"};
    command_keywords_["chmod"] = {"permission", "perms", "change", "izin", "chmod"};
    command_keywords_["chown"] = {"owner", "own", "sahip", "chown"};
    command_keywords_["tar"] = {"archive", "tar", "compress"};
    command_keywords_["zip"] = {"zip", "compress", "sıkıştır"};
    command_keywords_["nano"] = {"edit", "nano", "vi", "düzenle"};
    command_keywords_["vi"] = {"edit", "vi", "vim", "düzenle"};
    command_keywords_["pwd"] = {"path", "present", "working", "konum", "pwd"};
    command_keywords_["bash"] = {"bash", "shell", "terminal", "console", "kabuğu"};

    init_templates();
}

void ParameterExtractorV2::init_templates()
{
    command_templates_["cp"] = "cp <SRC> <DST>";
    command_templates_["mv"] = "mv <SRC> <DST>";
    command_templates_["rm"] = "rm <FILE>";
    command_templates_["mkdir"] = "mkdir <DIR>";
    command_templates_["cd"] = "cd <DIR>";
    command_templates_["ls"] = "ls <DIR>";
    command_templates_["cat"] = "cat <FILE>";
    command_templates_["grep"] = "grep <PATTERN> <FILE>";
    command_templates_["chmod"] = "chmod <PERMS> <FILE>";
    command_templates_["chown"] = "chown <OWNER> <FILE>";
    command_templates_["nano"] = "nano <FILE>";
    command_templates_["vi"] = "vi <FILE>";
    command_templates_["pwd"] = "pwd";
    command_templates_["bash"] = "bash";
}

string ParameterExtractorV2::normalize(const string &s) const
{
    string result = s;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

string ParameterExtractorV2::extract_command(const string &sentence)
{
    string sentence_lower = normalize(sentence);

    for (const auto &[cmd, keywords] : command_keywords_)
    {
        for (const auto &keyword : keywords)
        {
            if (sentence_lower.find(keyword) != string::npos)
            {
                return cmd;
            }
        }
    }

    return "";
}

string ParameterExtractorV2::extract_parameter(const string &sentence, const string &param_type)
{
    // Parameter extraction patterns
    vector<std::regex> patterns;

    if (param_type == "<FILE>")
    {
        patterns.push_back(std::regex(R"((?:file|dosya|document|doc|config)\s+([a-zA-Z0-9._-]+))", std::regex::icase));
        patterns.push_back(std::regex(R"(([a-zA-Z0-9._-]+\.(?:txt|py|csv|json|conf|log)))"));
    }
    else if (param_type == "<DIR>")
    {
        patterns.push_back(std::regex(R"((?:directory|folder|klasor|dizin)\s+([a-zA-Z0-9._/-]+))", std::regex::icase));
        patterns.push_back(std::regex(R"((?:to|in|den)\s+([a-zA-Z0-9._/-~]+)(?:\s|$))", std::regex::icase));
    }
    else if (param_type == "<SRC>")
    {
        patterns.push_back(std::regex(R"((?:copy|move|from|kaynak)\s+([a-zA-Z0-9._/-]+))", std::regex::icase));
        patterns.push_back(std::regex(R"(([a-zA-Z0-9._/-]+)\s+(?:to|e|ile))", std::regex::icase));
    }
    else if (param_type == "<DST>")
    {
        patterns.push_back(std::regex(R"((?:to|into|hedefe)\s+([a-zA-Z0-9._/-]+))", std::regex::icase));
        patterns.push_back(std::regex(R"((?:as|olarak)\s+([a-zA-Z0-9._/-]+))", std::regex::icase));
    }
    else if (param_type == "<PATTERN>")
    {
        patterns.push_back(std::regex(R"((?:search|find|ara)\s+(?:for\s+)?([a-zA-Z0-9.*+?^${}()|[\]\\-]+))", std::regex::icase));
        patterns.push_back(std::regex(R"("([^"]+)"));
    }
    else if (param_type == "<PERMS>")
    {
        patterns.push_back(std::regex(R"((?:permissions|perms|izin)\s+([0-7]{3}))", std::regex::icase));
    }

    // Try each pattern
    for (const auto &pattern : patterns)
    {
        std::smatch match;
        if (std::regex_search(sentence, match, pattern))
        {
            if (match.size() > 1)
            {
                return match[1].str();
            }
        }
    }

    return "";
}

unordered_map<string, string> ParameterExtractorV2::extract_parameters(
    const string &sentence,
    const string &command)
{
    unordered_map<string, string> params;

    if (command == "cp")
    {
        string src = extract_parameter(sentence, "<SRC>");
        string dst = extract_parameter(sentence, "<DST>");
        if (!src.empty())
            params["<SRC>"] = src;
        if (!dst.empty())
            params["<DST>"] = dst;
    }
    else if (command == "mv")
    {
        string src = extract_parameter(sentence, "<SRC>");
        string dst = extract_parameter(sentence, "<DST>");
        if (!src.empty())
            params["<SRC>"] = src;
        if (!dst.empty())
            params["<DST>"] = dst;
    }
    else if (command == "rm")
    {
        string file = extract_parameter(sentence, "<FILE>");
        if (!file.empty())
            params["<FILE>"] = file;
    }
    else if (command == "mkdir")
    {
        string dir = extract_parameter(sentence, "<DIR>");
        if (!dir.empty())
            params["<DIR>"] = dir;
    }
    else if (command == "cd")
    {
        string dir = extract_parameter(sentence, "<DIR>");
        if (!dir.empty())
            params["<DIR>"] = dir;
    }
    else if (command == "ls")
    {
        string dir = extract_parameter(sentence, "<DIR>");
        if (!dir.empty())
            params["<DIR>"] = dir;
        else
            params["<DIR>"] = ".";
    }
    else if (command == "cat")
    {
        string file = extract_parameter(sentence, "<FILE>");
        if (!file.empty())
            params["<FILE>"] = file;
    }
    else if (command == "grep")
    {
        string pattern = extract_parameter(sentence, "<PATTERN>");
        string file = extract_parameter(sentence, "<FILE>");
        if (!pattern.empty())
            params["<PATTERN>"] = pattern;
        if (!file.empty())
            params["<FILE>"] = file;
    }
    else if (command == "nano" || command == "vi")
    {
        string file = extract_parameter(sentence, "<FILE>");
        if (!file.empty())
            params["<FILE>"] = file;
    }

    return params;
}

string ParameterExtractorV2::fill_template(
    const string &template_str,
    const unordered_map<string, string> &params)
{
    string result = template_str;

    for (const auto &[tag, value] : params)
    {
        size_t pos = 0;
        while ((pos = result.find(tag, pos)) != string::npos)
        {
            result.replace(pos, tag.length(), value);
            pos += value.length();
        }
    }

    return result;
}

string ParameterExtractorV2::get_template(const string &command) const
{
    auto it = command_templates_.find(command);
    return it != command_templates_.end() ? it->second : "";
}

ParameterExtractorV2::ExtractedParams ParameterExtractorV2::process(const string &sentence)
{
    ExtractedParams result;
    result.success = false;
    result.command = "";
    result.filled_command = "";

    // Extract command
    result.command = extract_command(sentence);
    if (result.command.empty())
    {
        return result;
    }

    // Extract parameters
    result.parameters = extract_parameters(sentence, result.command);

    // Get template and fill it
    string template_str = get_template(result.command);
    if (template_str.empty())
    {
        result.filled_command = result.command;
    }
    else
    {
        result.filled_command = fill_template(template_str, result.parameters);
    }

    // Add <end> tag
    result.filled_command += " <end>";
    result.success = true;

    return result;
}
