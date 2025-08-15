import re

def parse_student_data(student_list):
    students = []
    for student in student_list.split(','):
        student_id, details = student.strip().split('[')
        rank, order = details[:-1].split('|')
        students.append({
            'student_id': student_id,
            'rank': int(rank),
            'order': int(order)
        })
    return students


def parse_file(file_path):
    data = {
        'steps': [],
        'schools': {},
        'students': {}
    }
    
    current_step = {}
    school_id = None
    pattern = re.compile(r'\[[X|\d+]\|\d+\]')

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            elif line == "===  SUMAR ===":
                break
            elif line.startswith('==='):
                # New cycle
                step_number = int(re.search(r'\d+', line).group())
                if current_step:
                    data['steps'].append(current_step)
                current_step = {'step_number': step_number, 'schools': {}}
            elif line.startswith('Obor:'):
                school_id = line.split(';')[0].split(':')[1].strip()
                if school_id not in data['schools']:
                    data['schools'][school_id] = {'capacity': 0, 'cycles': []}
            elif line.startswith('Zaplneni/Kapacita:'):
                filled, capacity = map(int, line.split(':')[1].strip().split('/'))
                data['schools'][school_id]['capacity'] = max(data['schools'][school_id]['capacity'], capacity)
                current_step['schools'][school_id] = {'filled': filled, 'capacity': capacity}
            elif 'Pridani' in line and pattern.search(line):
                num_added = int(re.search(r'\(\d+\)', line).group()[1:-1])
                student_list = line.split(':')[1].strip()
                students = parse_student_data(student_list)
                current_step['schools'][school_id]['added'] = students
                for student in students:
                    if student['student_id'] not in data['students']:
                        data['students'][student['student_id']] = []
                    data['students'][student['student_id']].append({
                        'school_id': school_id,
                        'cycle': current_step['step_number'],
                        'status': 'accepted',
                        'rank': student['rank'],
                        'order': student['order']
                    })
            elif 'Nevesli se' in line and pattern.search(line):
                student_list = line.split(':')[1].strip()
                students = parse_student_data(student_list)
                current_step['schools'][school_id]['not_fit'] = students
                for student in students:
                    if student['student_id'] not in data['students']:
                        data['students'][student['student_id']] = []
                    data['students'][student['student_id']].append({
                        'school_id': school_id,
                        'cycle': current_step['step_number'],
                        'status': 'moved',
                        'rank': student['rank'],
                        'order': student['order']
                    })
            elif 'Definitivne neprijati' in line and pattern.search(line):
                student_list = line.split(':')[1].strip()
                students = parse_student_data(student_list)
                current_step['schools'][school_id]['rejected'] = students
                for student in students:
                    if student['student_id'] not in data['students']:
                        data['students'][student['student_id']] = []
                    data['students'][student['student_id']].append({
                        'school_id': school_id,
                        'cycle': current_step['step_number'],
                        'status': 'rejected',
                        'rank': student['rank'],
                        'order': student['order']
                    })
            else:
                raise ValueError(f"Unrecognized line: {line}")
        if current_step:
            data['steps'].append(current_step)

    return data

data.keys()
data["steps"]
type(data["schools"])
len(data["schools"])
data

data

# Usage
file_path = "/home/thomas/Downloads/audit-log-prod.txt"
parsed_data = parse_file(file_path)

