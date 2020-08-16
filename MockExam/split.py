import numpy as np
import re
import time

def get_questions(filename):
    # read in the map
    inf = open(filename)
    questions = []
    question = None
    question_nums = set()
    for line in inf.readlines():
        if line.strip():
            q = re.search("[Qq]uestion\s*(\d+):(.*)", line)
            a = re.search("([a-zA-Z])\)(.*)", line)
            c = re.search("[Cc]orrect answer:\s*([a-zA-Z])", line)
            if q:
                # new question
                in_answer = False
                num = int(q.group(1))
                question = {
                    'num': num,
                    'question': f'{q.group(2).strip()}',
                    'answers': {},
                    'correct': None
                }
                questions.append(question)
                question_nums.add(num)
            elif a:
                # new answer
                answer = a.group(1).strip()
                value = a.group(2).strip()
                question['answers'][answer] = value
                in_answer = True
            elif c:
                # correct answer
                correct = c.group(1)
                question['correct'] = correct
            else:
                if in_answer:
                    question['answers'][answer] += f'\n{line.strip()}'
                else:
                    question['question'] += f'\n{line.strip()}'
    return question_nums, questions


def print_question(question):
    print()
    print(f'Question {question["num"]}')
    print(question['question'])
    print()
    for a, answer in question['answers'].items():
        print(f'{a}) {answer}')


def quiz(questions, quiz_size=20):
    questions_len = len(questions)
    sampling = np.random.choice(len(questions), quiz_size, replace=False)
    questions = [questions[q] for q in sampling]
    correct = []
    for q in questions:
        print_question(q)
        print()
        ans = input('Answer (q to quit): ')
        if ans.lower() == 'q':
            break
        elif ans.upper() == q['correct']:
            print('Correct!')
            correct.append(q)
        else:
            print(f'Incorrect. Correct answer is {q["correct"]}')
    return questions, correct


if __name__=="__main__":
    filename = 'exam.txt'
    question_nums, questions = get_questions(filename)

    max_question = max(question_nums)
    print(f'Processed {max_question} Questions')
    all_question_nums = set(range(1, max_question+1))
    missing_question_nums = all_question_nums.difference(question_nums)
    if len(missing_question_nums) > 0:
        print(f'Missing questions: {"".join(missing_question_nums)}')

    quiz_size = 5
    start = time.time()
    chosen_questions, correct = quiz(questions, quiz_size)
    end = time.time()
    print(f'Quiz (size={quiz_size}) finished. {len(correct)}/{len(chosen_questions)} correct. Grade: {round(len(correct)/len(chosen_questions), 2)}. Time: {round(end - start, 2)} seconds.')
