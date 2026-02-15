/**
 * @file ParameterExtractorV2.h
 * @brief Enhanced parameter extraction with template substitution
 *        Integrates <SRC>, <DST>, <FILE>, <DIR>, <PATTERN>, <PERMS>, <OWNER> tags
 */

#ifndef PARAMETER_EXTRACTOR_V2_H
#define PARAMETER_EXTRACTOR_V2_H

#include <string>
#include <vector>
#include <unordered_map>
#include <regex>
#include <algorithm>
#include <iostream>

using std::string;
using std::vector;
using std::unordered_map;

class ParameterExtractorV2
{
public:
    /**
     * @struct ExtractedParams
     * @brief Container for extracted parameters
     */
    struct ExtractedParams
    {
        string command;
        unordered_map<string, string> parameters; // <TAG> -> value
        string filled_command;                     // Command with parameters replaced
        bool success;
    };

    ParameterExtractorV2()
    {
        init_command_keywords();
    }

    /**
     * @brief Extract command from user sentence
     * @param sentence User input (e.g., "copy backup to projects")
     * @return Command name (e.g., "cp")
     */
    string extract_command(const string &sentence);

    /**
     * @brief Extract parameters for specific command
     * @param sentence User input
     * @param command Command name (e.g., "cp")
     * @return Map of <TAG> -> value
     */
    unordered_map<string, string> extract_parameters(
        const string &sentence,
        const string &command);

    /**
     * @brief Fill command template with parameters
     * @param template_str Command template (e.g., "cp <SRC> <DST>")
     * @param params Parameters map
     * @return Filled command (e.g., "cp backup projects")
     */
    string fill_template(
        const string &template_str,
        const unordered_map<string, string> &params);

    /**
     * @brief Full extraction pipeline
     * @param sentence User input
     * @return ExtractedParams with command, params, and filled command
     */
    ExtractedParams process(const string &sentence);

    /**
     * @brief Get template for command
     * @param command Command name
     * @return Template (e.g., "cp <SRC> <DST>")
     */
    string get_template(const string &command) const;

private:
    unordered_map<string, vector<string>> command_keywords_;
    unordered_map<string, string> command_templates_;

    void init_command_keywords();
    void init_templates();

    /**
     * @brief Extract single parameter using regex
     * @param sentence User input
     * @param param_type Parameter type (e.g., "<FILE>")
     * @return Extracted value or empty string
     */
    string extract_parameter(const string &sentence, const string &param_type);

    /**
     * @brief Normalize sentence for matching
     */
    string normalize(const string &s) const;
};

#endif // PARAMETER_EXTRACTOR_V2_H
