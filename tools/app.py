import json
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()
JSON_FILE = 'data.json'

# ========================================================
# 3 TOOLS DEFINITION
# ========================================================
def timetable_analysis(day: str) -> str:
    """
    Analyzes a raw timetable string to find classes for a specific day.
    Args:
        day: The day of the week to check (e.g., 'Monday', 'Tuesday').
    """
    with open(JSON_FILE, 'r') as file:
        data = json.load(file)
    day_key = day.strip().capitalize()
    schedule = data.get("timetable_data", {}).get(day_key, [])
    if not schedule:
        return f"No classes scheduled for {day_key}."
    result = f"Classes for {day_key}:\n"
    for item in schedule:
        result += f"- {item['time']}: {item['subject']} in {item['room']}\n"
    return result

def calculate_attendance_percentage(subject_code: str) -> str:
    """
    Calculates the exact attendance percentage and warning status for a course.
    Args:
        subject_code: The exact course code string (e.g., 'CS101', 'MA102').
    """
    with open(JSON_FILE, 'r') as file:
        data = json.load(file)
    code = subject_code.upper().strip()
    stats = data.get("attendance_stats", {}).get(code)
    if not stats:
        return f"Error: Subject '{subject_code}' not found."
    pct = (stats["attended"] / stats["total"]) * 100
    status = "Safe" if pct >= 75 else "Low Attendance Warning"
    return f"{stats['subject_name']} ({code}) Attendance is {pct:.2f}%. Status: {status}."

def mark_attendance(subject_code: str, status: str) -> str:
    """
    Marks a student as Present or Absent for a class and updates data.json.
    Args:
        subject_code: The code of the course (e.g., 'CS101', 'MA102').
        status: The entry status, strictly either 'Present' or 'Absent'.
    """
    with open(JSON_FILE, 'r') as file:
        data = json.load(file)
    code = subject_code.upper().strip()
    if code not in data["attendance_stats"]:
        return f"Error: Subject '{subject_code}' does not exist."
    stats = data["attendance_stats"][code]
    stats["total"] += 1
    if status.lower() == "present":
        stats["attended"] += 1
    with open(JSON_FILE, 'w') as file:
        json.dump(data, file, indent=4)
    return f"Logged: {code} marked as {status}. New total: {stats['attended']}/{stats['total']}"

# ========================================================
# MAIN SESSION LOOP
# ========================================================
if __name__ == "__main__":
    tools_map = {
        "timetable_analysis": timetable_analysis,
        "calculate_attendance_percentage": calculate_attendance_percentage,
        "mark_attendance": mark_attendance
    }
    
    print("The assistant can help you with your timetable analysis, attendance calculation, or marking.")
    print("Type 'exit' to quit the program.\n")

    while True:
        try:
            user_query = input("Ask the College Assistant: ").strip()
            if user_query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            if not user_query:
                continue

            with open(JSON_FILE, 'r') as file:
                data = json.load(file)
                
            prompt = (
                f"User Question: {user_query}\n\n"
                f"Context Database:\n"
                f"Timetable Details: {json.dumps(data.get('timetable_data'))}\n"
                f"Attendance Statistics: {json.dumps(data.get('attendance_stats'))}"
            )
            
            # Step 1: Request tool selection with manual execution handling
            response = client.models.generate_content(
                model='gemini-3.5-flash-lite',
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=list(tools_map.values()),
                    system_instruction=(
                        "You are an automated coordinator. You MUST choose and call one of the "
                        "provided tools when a user asks to mark attendance, analyze schedules, or calculate metrics."
                    ),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )
            
            tool_called = False
            
            # Step 2: Use candidates[0] directly to inspect the model's output parts
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    # Check if the part contains a function call request
                    if part.function_call:
                        tool_name = part.function_call.name
                        tool_args = part.function_call.args
                        
                        if tool_name in tools_map:
                            tool_called = True
                            print(f"🔧 [Tool Used]: {tool_name}")
                            
                            # Execute the matching python function locally
                            tool_result = tools_map[tool_name](**tool_args)
                            
                            # Step 3: Bundle the result into a clean text prompt for the final message
                            final_prompt = (
                                f"{prompt}\n\n"
                                f"[System Update: The tool '{tool_name}' was executed successfully.\n"
                                f"Result returned from local database]:\n{tool_result}\n\n"
                                f"Please provide the clean final answer text for the student based on this result."
                            )
                            
                            final_response = client.models.generate_content(
                                model='gemini-3.5-flash-lite',
                                contents=final_prompt
                            )
                            print(f"AI Output: {final_response.text}\n")
                            break
            
            if not tool_called:
                print("🔧 [Tool Used]: None (Direct Answer)")
                print(f"AI Output: {response.text}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"System Error encountered: {str(e)}\n")