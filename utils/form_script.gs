// The Google side of the form import: creates the form and its spreadsheet, and handles each submission.
// It is kept here so it can be reviewed; docs/form-import.md explains how to install it. Nothing here runs in CI.

const FAILURE_MODES = [
  "correctness",
  "correct_tool_calls",
  "code_safety",
  "robustness_against_adversarial_inputs",
  "sycophancy",
  "planning",
  "reasoning",
  "uncertainty_awareness",
  "aesthetic_quality",
  "lost_context_on_multi_agent_tasks",
  "implicit_domain_knowledge"
];

const VALIDATORS = [
  "judge:rubric",
  "content:numeric_match",
  "content:json_output",
  "content:numeric_within_range",
  "content:paper_json_match",
  "content:scientific_notation",
  "content:exact_str_match",
  "content:contains_str",
  "format:json_output"
];

const MAX_CHECKS = 4;
const MAX_SOURCES = 3;
const FIRST_TASK_NUMBER = 52;
const RESPONSES_TAB = "Responses";
const AUTHORS_TAB = "Authors";
const AUTHOR_COLUMNS = ["Task ID", "Full name", "Email", "Institution", "ORCID", "OpenReview"];

// CREDIT and RETENTION are still to be decided with the maintainers, and must be filled in before the form goes live.
const PRIVACY_NOTE =
  "What we store: your name, email, institution, ORCID and OpenReview handle, in a private spreadsheet. " +
  "Uploading a file also records your Google account name, which we remove from the file before publishing. " +
  "Who can see it: only the SciFlawBench maintainers. We use it to contact you and to credit you, and it is never " +
  "published with a task. What is published: if your task is accepted, its text, checks, sources and files go into " +
  "the public SciFlawBench repository. Credit: CREDIT. How long we keep it: RETENTION.";

const PROMPT_HELP = `The exact prompt the agent gets. If a check reads the answer automatically, say in the prompt exactly how the agent must write its final answer, with no text, units or code block around it:
- One value: only that value, such as 12.3
- Several values: a JSON object, such as {"mass": 12.3, "spin": 0.7}. A JSON check grades every field exactly, so state the rounding. Put any explanation in a field such as "justification".
To use a file, write data/tasks/assets/{TASK_ID}/data.csv`;

const ANSWER_HELP =
  "A number for numeric checks, written like 6.022e23 for scientific notation. min, max for a range. " +
  "Otherwise text, a JSON object, or arXiv ids. Leave it empty for a rubric.";

// Run once by hand. Afterwards add the File upload question and publish the form, as the docs describe.
function setup() {
  const form = FormApp.create("SciFlawBench task submission");
  form.setDescription(PRIVACY_NOTE);

  form.addTextItem().setTitle("Full name").setRequired(true);
  form
    .addTextItem()
    .setTitle("Email")
    .setRequired(true)
    .setValidation(FormApp.createTextValidation().requireTextIsEmail().build());
  form.addTextItem().setTitle("Institution").setRequired(true);
  form
    .addTextItem()
    .setTitle("ORCID")
    .setRequired(true)
    .setHelpText("Like 0000-0002-1825-0097")
    .setValidation(
      FormApp.createTextValidation()
        .requireTextMatchesPattern("\\d{4}-\\d{4}-\\d{4}-\\d{3}[\\dX]")
        .setHelpText("Write it like 0000-0002-1825-0097")
        .build()
    );
  form.addTextItem().setTitle("OpenReview");

  form.addPageBreakItem().setTitle("The task");
  form
    .addParagraphTextItem()
    .setTitle("Task description")
    .setRequired(true)
    .setHelpText(PROMPT_HELP);

  const addAnother = [];
  const checkPages = [];
  for (let n = 1; n <= MAX_CHECKS; n++) {
    checkPages.push(form.addPageBreakItem().setTitle(`Check ${n}`));
    form.addListItem().setTitle(`Check ${n}: failure mode`).setChoiceValues(FAILURE_MODES).setRequired(true);
    form.addListItem().setTitle(`Check ${n}: validator`).setChoiceValues(VALIDATORS).setRequired(true);
    form.addParagraphTextItem().setTitle(`Check ${n}: correct answer`).setHelpText(ANSWER_HELP);
    form
      .addTextItem()
      .setTitle(`Check ${n}: tolerance`)
      .setHelpText("Numeric checks only. Relative for scientific notation, otherwise absolute.")
      .setValidation(FormApp.createTextValidation().requireNumber().build());
    form.addParagraphTextItem().setTitle(`Check ${n}: rubric`).setHelpText("Rubric checks only.");
    if (n < MAX_CHECKS) {
      addAnother.push(form.addMultipleChoiceItem().setTitle(`Check ${n}: add another check?`).setRequired(true));
    }
  }

  const sourcesPage = form.addPageBreakItem().setTitle("Sources and files");
  for (let n = 1; n <= MAX_SOURCES; n++) {
    form
      .addTextItem()
      .setTitle(`Source ${n}: URL`)
      .setValidation(FormApp.createTextValidation().requireTextIsUrl().build());
    form.addTextItem().setTitle(`Source ${n}: note`);
  }

  addAnother.forEach((item, index) => {
    item.setChoices([item.createChoice("Yes", checkPages[index + 1]), item.createChoice("No", sourcesPage)]);
  });

  const spreadsheet = SpreadsheetApp.create("SciFlawBench task responses");
  form.setDestination(FormApp.DestinationType.SPREADSHEET, spreadsheet.getId());
  const tab = linkedTab(spreadsheet);
  tab.setName(RESPONSES_TAB);
  tab.getRange(1, tab.getLastColumn() + 1).setValue("Task ID");
  // keep answers exactly as typed: otherwise Sheets turns 1/2 into a date and 007 into 7
  tab.getRange(1, 1, tab.getMaxRows(), tab.getMaxColumns()).setNumberFormat("@");
  spreadsheet.insertSheet(AUTHORS_TAB).appendRow(AUTHOR_COLUMNS);

  ScriptApp.newTrigger("onFormSubmit").forSpreadsheet(spreadsheet).onFormSubmit().create();
  Logger.log("Form to edit: " + form.getEditUrl());
  Logger.log("Spreadsheet: " + spreadsheet.getUrl());
}

function linkedTab(spreadsheet) {
  for (let attempt = 0; attempt < 30; attempt++) {
    SpreadsheetApp.flush();
    const tab = spreadsheet.getSheets().find((sheet) => sheet.getFormUrl());
    if (tab) return tab;
    Utilities.sleep(1000);
  }
  throw new Error("The form's response tab did not appear. Delete the form and the spreadsheet and run setup again.");
}

function nextTaskId() {
  const properties = PropertiesService.getScriptProperties();
  const number = Number(properties.getProperty("NEXT_TASK_NUMBER") || FIRST_TASK_NUMBER);
  properties.setProperty("NEXT_TASK_NUMBER", String(number + 1));
  return "SFB-" + String(number).padStart(4, "0");
}

// Runs on each submission: gives it the next id, records the author privately, and sends the emails.
function onFormSubmit(e) {
  const tab = e.range.getSheet();
  const header = tab.getRange(1, 1, 1, tab.getLastColumn()).getDisplayValues()[0];
  const answers = tab.getRange(e.range.getRow(), 1, 1, tab.getLastColumn()).getDisplayValues()[0];
  const answer = (title) => answers[header.indexOf(title)];

  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  let taskId;
  try {
    taskId = nextTaskId();
    tab.getRange(e.range.getRow(), header.indexOf("Task ID") + 1).setValue(taskId);
    const row = AUTHOR_COLUMNS.map((title) => (title === "Task ID" ? taskId : answer(title)));
    e.source.getSheetByName(AUTHORS_TAB).appendRow(row);
  } finally {
    lock.releaseLock();
  }

  const maintainers = PropertiesService.getScriptProperties().getProperty("MAINTAINERS");
  if (maintainers) {
    MailApp.sendEmail(maintainers, `New SciFlawBench submission: ${taskId}`, "It is ready to import.");
  }
  MailApp.sendEmail(
    answer("Email"),
    `Your SciFlawBench task: ${taskId}`,
    `Thank you for your submission. Its task id is ${taskId}. We will review it and contact you if we have questions.`
  );
}
