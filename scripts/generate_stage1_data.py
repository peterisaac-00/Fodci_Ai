import os
from pathlib import Path

DATA_DIR = Path("/home/ubuntu/backend-ai/training_data/fundamentals")

EXAMPLES = [
    {
        "instruction": "Explain what a Backend is in the context of web development.",
        "input": "",
        "response": "The backend (or server-side) of a web application is the part that users don't see. It is responsible for storing and organizing data, and ensuring everything on the client-side (frontend) actually works. The backend communicates with the frontend, sending and receiving information to be displayed as a web page. It typically consists of a server, an application, and a database."
    },
    {
        "instruction": "What are the primary components of a typical Backend architecture?",
        "input": "",
        "response": "A typical backend architecture consists of three main parts:\n1. The Server: The computer that receives requests.\n2. The App: The application running on the server that processes requests and contains the business logic.\n3. The Database: Where data is persistently stored and retrieved."
    },
    {
        "instruction": "Define HTTP and its role in Backend Engineering.",
        "input": "",
        "response": "HTTP (Hypertext Transfer Protocol) is the foundation of data communication for the World Wide Web. In backend engineering, it is the primary protocol used for communication between the client (frontend) and the server (backend). It defines how messages are formatted and transmitted, and what actions web servers and browsers should take in response to various commands."
    },
    {
        "instruction": "Explain the difference between an HTTP Request and an HTTP Response.",
        "input": "",
        "response": "An HTTP Request is sent by the client (e.g., a browser) to the server to initiate an action (like fetching a page or submitting a form). An HTTP Response is sent by the server back to the client, containing the result of the request (like the requested HTML, an image, or a JSON object) and a status code indicating success or failure."
    },
    {
        "instruction": "What is a REST API?",
        "input": "",
        "response": "REST (Representational State Transfer) is an architectural style for designing networked applications. A REST API is an application programming interface that uses HTTP requests to GET, PUT, POST, and DELETE data. It is based on resources, which are identified by URLs, and typically uses JSON for data exchange."
    },
    {
        "instruction": "List common HTTP methods and their intended use in a RESTful API.",
        "input": "",
        "response": "The common HTTP methods are:\n- GET: Retrieve a resource or a collection of resources.\n- POST: Create a new resource.\n- PUT: Update an existing resource (full replacement).\n- PATCH: Partially update an existing resource.\n- DELETE: Remove a resource."
    },
    {
        "instruction": "Explain HTTP Status Code 200 OK.",
        "input": "",
        "response": "The HTTP 200 OK status code indicates that the request has succeeded. The information returned with the response is dependent on the method used in the request (e.g., the entity corresponding to the requested resource for GET, or an entity describing the result of the action for POST)."
    },
    {
        "instruction": "Explain HTTP Status Code 201 Created.",
        "input": "",
        "response": "The HTTP 201 Created status code indicates that the request has been fulfilled and has resulted in one or more new resources being created. This is the standard response for a successful POST request."
    },
    {
        "instruction": "Explain HTTP Status Code 400 Bad Request.",
        "input": "",
        "response": "The HTTP 400 Bad Request status code indicates that the server cannot or will not process the request due to something that is perceived to be a client error (e.g., malformed request syntax, invalid request message framing, or deceptive request routing)."
    },
    {
        "instruction": "Explain HTTP Status Code 401 Unauthorized.",
        "input": "",
        "response": "The HTTP 401 Unauthorized status code indicates that the request has not been applied because it lacks valid authentication credentials for the target resource. It means 'I don't know who you are, or your credentials are invalid'."
    },
    {
        "instruction": "Explain HTTP Status Code 403 Forbidden.",
        "input": "",
        "response": "The HTTP 403 Forbidden status code indicates that the server understood the request but refuses to authorize it. Unlike 401, the client's identity is known to the server, but they do not have permission to access the resource."
    },
    {
        "instruction": "Explain HTTP Status Code 404 Not Found.",
        "input": "",
        "response": "The HTTP 404 Not Found status code indicates that the origin server did not find a current representation for the target resource or is not willing to disclose that one exists. This is commonly seen when a URL is mistyped or a resource has been deleted."
    },
    {
        "instruction": "Explain HTTP Status Code 500 Internal Server Error.",
        "input": "",
        "response": "The HTTP 500 Internal Server Error status code indicates that the server encountered an unexpected condition that prevented it from fulfilling the request. This is a generic 'catch-all' error for server-side crashes or unhandled exceptions."
    },
    {
        "instruction": "What is JSON and why is it popular in Backend development?",
        "input": "",
        "response": "JSON (JavaScript Object Notation) is a lightweight data-interchange format. It is easy for humans to read and write and easy for machines to parse and generate. It is popular in backend development because it is language-independent, highly portable, and serves as the standard format for data exchange in modern web APIs."
    },
    {
        "instruction": "Describe the typical flow of a Backend request.",
        "input": "",
        "response": "The typical flow is:\n1. Client sends an HTTP Request to a specific URL.\n2. The Web Server receives the request and routes it to the Application.\n3. The Application processes the request (Authentication, Validation, Business Logic).\n4. The Application interacts with the Database if needed.\n5. The Application constructs an HTTP Response (e.g., JSON data + status code).\n6. The Web Server sends the response back to the Client."
    },
    {
        "instruction": "Why should you use POST instead of GET for creating a user?",
        "input": "A developer wants to create a new user account by sending the username and password.",
        "response": "You should use POST because:\n1. Semantics: GET is intended for retrieving data, while POST is for creating/modifying data.\n2. Security: GET parameters are visible in the URL, browser history, and server logs, which is unsafe for sensitive data like passwords. POST sends data in the request body.\n3. Data Limits: URLs have length limits, while the POST body can handle much larger amounts of data."
    },
    {
        "instruction": "What is the difference between Authentication and Authorization?",
        "input": "",
        "response": "Authentication is the process of verifying who a user is (e.g., checking a username and password). Authorization is the process of verifying what a user has access to (e.g., checking if a logged-in user is an administrator allowed to delete a resource)."
    },
    {
        "instruction": "What is a Middleware in a Backend application?",
        "input": "",
        "response": "Middleware is software that acts as a bridge between an operating system or database and applications, especially on a network. In backend frameworks, middleware functions are functions that have access to the request object, the response object, and the next middleware function in the application’s request-response cycle. They are used for tasks like logging, authentication, and parsing request bodies."
    },
    {
        "instruction": "Explain the concept of Statelessness in REST APIs.",
        "input": "",
        "response": "Statelessness means that the server does not store any state about the client session on the server-side. Each request from the client to the server must contain all the information necessary to understand and complete the request. This improves scalability because the server doesn't need to remember previous requests."
    },
    {
        "instruction": "What is an API Endpoint?",
        "input": "",
        "response": "An API endpoint is a specific URL where an API can be accessed by a client application. It represents a specific resource or function provided by the API. For example, 'https://api.example.com/v1/users' might be an endpoint to retrieve or manage user data."
    }
]

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for i, ex in enumerate(EXAMPLES):
        filename = f"example_{i:03d}.txt"
        input_text = ex['input'] if ex['input'].strip() else "None"
        content = (
            f"### Instruction\n{ex['instruction']}\n\n"
            f"### Input\n{input_text}\n\n"
            f"### Response\n{ex['response']}\n"
        )
        with open(DATA_DIR / filename, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"Generated {len(EXAMPLES)} examples in {DATA_DIR}")

if __name__ == "__main__":
    main()
