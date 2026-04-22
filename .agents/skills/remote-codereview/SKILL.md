---
name: remote-codereview
description: Review a remote GitHub PR or GitLab MR for high-signal bugs, regressions, correctness issues, risky assumptions, and scoped guideline violations.
---
Provide a code review for the given pull request or merge request.

**Agent assumptions (applies to all agents and subagents):**
- All tools are functional and will work without error. Do not test tools or make exploratory calls. Make sure this is clear to every subagent that is launched.
- Only call a tool if it is required to complete the task. Every tool call should have a clear purpose.

To do this, follow these steps precisely:

1. Launch a haiku agent to check if any of the following are true:
   - The pull request is closed
   - The pull request is a draft
   - The pull request does not need code review (e.g. automated PR, trivial change that is obviously correct)
   - Claude has already commented on this review target (use the platform-appropriate command: `gh pr view <PR> --comments` for GitHub PRs, `glab mr view <MR> --comments` for GitLab MRs)

   If any condition is true, stop and do not proceed.

Note: Still review Claude generated PR's.

2. Launch a haiku agent to return a list of file paths (not their contents) for all relevant repository guidance files including:
   - The root `AGENTS.md` file, if it exists
   - Any directory-scoped `AGENTS.md` files that apply to modified files
   - Any files under `.agents/instructions/` or `.agents/skills/` referenced by those `AGENTS.md` files and relevant to the review scope

3. Launch a sonnet agent to view the pull request and return a summary of the changes

4. Launch 4 agents in parallel to independently review the changes. Each agent should return the list of issues, where each issue includes a description and the reason it was flagged (e.g. "guidance adherence", "bug"). The agents should do the following:

   Agents 1 + 2: repository guidance compliance sonnet agents
   Audit changes for `AGENTS.md` / `.agents` guidance compliance in parallel. Note: When evaluating guidance compliance for a file, only consider the guidance files that are in scope for that file, including applicable parent `AGENTS.md` files and the `.agents` instruction/skill files they reference.

   Agent 3: Opus bug agent (parallel subagent with agent 4)
   Scan for obvious bugs. Focus only on the diff itself without reading extra context. Flag only significant bugs; ignore nitpicks and likely false positives. Do not flag issues that you cannot validate without looking at context outside of the git diff.

   Agent 4: Opus bug agent (parallel subagent with agent 3)
   Look for problems that exist in the introduced code. This could be security issues, incorrect logic, etc. Only look for issues that fall within the changed code.

   **CRITICAL: We only want HIGH SIGNAL issues.** Flag issues where:
   - The code will fail to compile or parse (syntax errors, type errors, missing imports, unresolved references)
   - The code will definitely produce wrong results regardless of inputs (clear logic errors)
   - Clear, unambiguous repository guidance violations where you can quote the exact rule being broken

   Do NOT flag:
   - Code style or quality concerns
   - Potential issues that depend on specific inputs or state
   - Subjective suggestions or improvements

   If you are not certain an issue is real, do not flag it. False positives erode trust and waste reviewer time.

   In addition to the above, each subagent should be told the PR title and description. This will help provide context regarding the author's intent.

5. For each issue found in the previous step by agents 3 and 4, launch parallel subagents to validate the issue. These subagents should get the PR title and description along with a description of the issue. The agent's job is to review the issue to validate that the stated issue is truly an issue with high confidence. For example, if an issue such as "variable is not defined" was flagged, the subagent's job would be to validate that is actually true in the code. Another example would be repository guidance issues. The agent should validate that the cited `AGENTS.md` / `.agents` rule is in scope for this file and is actually violated. Use Opus subagents for bugs and logic issues, and sonnet agents for guidance violations.

6. Filter out any issues that were not validated in step 5. De-duplicate overlapping issues across all reviewers, then assign a final severity to each remaining issue. This step will give us our list of high signal issues for our review.

7. Output a summary of the review findings to the terminal:
   - Use the standardized report structure in `REPORT_TEMPLATE.md`.
   - If issues were found, fill in the severity sections with only validated, de-duplicated high-signal issues.
   - Number issues sequentially across severity sections.
   - If no issues were found, use the exact text: "No issues found. Checked for bugs and scoped guidance compliance." in the `Findings` section.

   If `--comment` argument was NOT provided, stop here. Do not post any review comments.

   If `--comment` argument IS provided and NO issues were found, post a summary comment using the same report structure and stop.

   If `--comment` argument IS provided and issues were found, continue to step 8.

8. Create a list of all comments that you plan on leaving. This is only for you to make sure you are comfortable with the comments. Do not post this list anywhere.

9. Post inline comments for each issue using the platform-appropriate mechanism. For each comment:
   - For GitHub PRs, use the GitHub inline comment tool with `confirmed: true`.
   - For GitLab MRs, use the available GitLab review or note tooling. If inline comments are not available in the current toolset, post the top-level summary comment only and explicitly note that inline comments were skipped due to tool limitations.
   - Provide a brief description of the issue.
   - For small, self-contained fixes, include a committable suggestion block only when the platform supports it.
   - For larger fixes (6+ lines, structural changes, or changes spanning multiple locations), describe the issue and suggested fix without a suggestion block.
   - Never post a committable suggestion UNLESS committing the suggestion fixes the issue entirely. If follow up steps are required, do not leave a committable suggestion.

   **IMPORTANT: Only post ONE comment per unique issue. Do not post duplicate comments.**

Use this list when evaluating issues in Steps 4 and 5 (these are false positives, do NOT flag):

- Pre-existing issues
- Something that appears to be a bug but is actually correct
- Pedantic nitpicks that a senior engineer would not flag
- Issues that a linter will catch (do not run the linter to verify)
- General code quality concerns (e.g., lack of test coverage, general security issues) unless explicitly required in the applicable repository guidance
- Issues mentioned in the applicable repository guidance but explicitly silenced in the code (e.g., via a lint ignore comment)

Notes:

- Use platform-appropriate CLI tools for the review target (`gh` for GitHub PRs, `glab` for GitLab MRs). Do not use web fetch.
- Create a todo list before starting.
- You must cite and link each issue in inline comments or summary comments (e.g., if referring to `AGENTS.md` or a `.agents` guidance file, include a link to it).
- Use the report template in `REPORT_TEMPLATE.md` for both terminal summaries and top-level PR/MR summary comments.
- If no issues are found and `--comment` argument is provided, the comment must still preserve the following exact content inside the `Findings` section:

---

## Code review

No issues found. Checked for bugs and scoped guidance compliance.

---

- When linking to code or guidance in comments, use the canonical web URL format for the target platform and commit SHA when required by that platform's Markdown renderer.
   - For GitHub PR comments, use links like `https://github.com/<owner>/<repo>/blob/<full_sha>/<path>#L10-L15`.
   - For GitLab MR comments, use the canonical GitLab blob URL for the relevant commit SHA and line range when available.
   - Always include enough line context to make the reference unambiguous.
