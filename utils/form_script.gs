// The Google side of the form import: gives each submission its task id, records its author privately, and sends the
// emails. The form itself is built and edited by hand in Google Forms; this script lives in its response spreadsheet.
// It is kept here so it can be reviewed; docs/form-import.md explains how to install it. Nothing here runs in CI.

const FIRST_TASK_NUMBER = 52;
const RESPONSES_TAB = "Responses";
const AUTHORS_TAB = "Authors";
const TASK_ID = "Task ID";
// "Task ID", then the form's questions about the author, by their titles on the form; case and spacing don't matter.
const AUTHOR_COLUMNS = ["Task ID", "Full Name", "Email", "Institution", "ORCID", "OpenReview"];

// Run once by hand, from the response spreadsheet (Extensions → Apps Script). Running it again changes nothing.
function attach() {
  const spreadsheet = SpreadsheetApp.getActiveSpreadsheet();
  const tab = spreadsheet.getSheets().find((sheet) => sheet.getFormUrl());
  if (!tab) {
    throw new Error("No tab here is linked to a form. In the form, open Responses → Link to Sheets, and pick this one.");
  }
  tab.setName(RESPONSES_TAB);
  if (columnOf(tab, TASK_ID) < 0) {
    tab.getRange(1, tab.getLastColumn() + 1).setValue(TASK_ID);
  }
  // keep answers exactly as typed: otherwise Sheets turns 1/2 into a date and 007 into 7
  tab.getRange(1, 1, tab.getMaxRows(), tab.getMaxColumns()).setNumberFormat("@");
  if (!spreadsheet.getSheetByName(AUTHORS_TAB)) {
    spreadsheet.insertSheet(AUTHORS_TAB).appendRow(AUTHOR_COLUMNS);
  }
  const installed = ScriptApp.getProjectTriggers().some((trigger) => trigger.getHandlerFunction() === "onFormSubmit");
  if (!installed) {
    ScriptApp.newTrigger("onFormSubmit").forSpreadsheet(spreadsheet).onFormSubmit().create();
  }
  Logger.log("Attached to " + spreadsheet.getUrl());
}

function titleKey(title) {
  return String(title).trim().replace(/\s+/g, " ").toLowerCase();
}

// The 0-based position of a column from its title, or -1.
function columnOf(tab, title) {
  const header = tab.getRange(1, 1, 1, tab.getLastColumn()).getDisplayValues()[0];
  return header.map(titleKey).indexOf(titleKey(title));
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
  const header = tab.getRange(1, 1, 1, tab.getLastColumn()).getDisplayValues()[0].map(titleKey);
  const answers = tab.getRange(e.range.getRow(), 1, 1, tab.getLastColumn()).getDisplayValues()[0];
  const answer = (title) => answers[header.indexOf(titleKey(title))] || "";

  const lock = LockService.getScriptLock();
  lock.waitLock(30000);
  let taskId;
  try {
    taskId = nextTaskId();
    tab.getRange(e.range.getRow(), header.indexOf(titleKey(TASK_ID)) + 1).setValue(taskId);
    const row = AUTHOR_COLUMNS.map((title) => (title === TASK_ID ? taskId : answer(title)));
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
