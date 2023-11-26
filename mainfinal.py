import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', config='bert-base-uncased')

# it read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


context = read_text_from_file('kangaroo.txt')

# QA dataset
qa_dataset = [
    {
        'context': "Kangaroos are marsupials that are native to Australia. They are known for their powerful hind legs, strong tails, and distinctive hopping motion. There are several species of kangaroos, including the red kangaroo, eastern grey kangaroo, and western grey kangaroo. Kangaroos are herbivores and mainly eat grasses and other vegetation. They have a unique reproductive system where females have a pouch in which they carry and nurse their young, called joeys.",
        'question': "What do kangaroos eat?",
        'answer': "Kangaroos are herbivores and mainly eat grasses and other vegetation."
    },
    {
        'context': "Kangaroos are fascinating creatures with incredible adaptations. They have powerful hind legs that allow them to cover large distances with impressive leaps. Kangaroos are social animals and often form groups known as mobs. The red kangaroo, the largest marsupial, is known for its distinctive reddish-brown fur and can reach speeds of up to 56 kilometers per hour.",
        'question': "How fast can a red kangaroo run?",
        'answer': "The red kangaroo can reach speeds of up to 56 kilometers per hour."
    },
    {
        'context': "The life cycle of a kangaroo is truly unique. Female kangaroos give birth to relatively undeveloped young, known as joeys. The tiny joeys, often no larger than a lima bean, continue their development in the mother's pouch. As they grow, joeys gradually spend more time outside the pouch but continue to nurse for an extended period.",
        'question': "How do kangaroos raise their young?",
        'answer': "Female kangaroos carry and nurse their young, called joeys, in their pouch. Joeys continue to nurse and develop outside the pouch as they grow."
    },
    {
        'context': "Kangaroos play a crucial role in the ecosystem by controlling vegetation through their grazing habits. Their unique digestive system allows them to efficiently extract nutrients from tough grasses. Kangaroos are also important culturally, often appearing in Aboriginal Australian stories and art as symbols of strength and adaptability.",
        'question': "What role do kangaroos play in the ecosystem?",
        'answer': "Kangaroos play a crucial role in the ecosystem by controlling vegetation through their grazing habits."
    },
    {
        'context': "The conservation of kangaroos is of great importance. While they are iconic symbols of Australia, some species face threats such as habitat loss and conflicts with human activities. Conservation efforts focus on preserving their natural habitats and addressing challenges to ensure the continued survival of these unique marsupials.",
        'question': "Why is the conservation of kangaroos important?",
        'answer': "Conservation efforts are important to address threats such as habitat loss and conflicts with human activities, ensuring the continued survival of kangaroo species."
    },
]

max_length = 512
def prepare_qa_data(data):
    qa_inputs = []

    for example in data:
        inputs = tokenizer(
            example['question'],
            example['context'],
            truncation='only_second',
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )

        start_positions = tokenizer(
            example['answer'],
            example['context'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )['input_ids'].view(-1).nonzero().squeeze().tolist()[0]

        end_positions = tokenizer(
            example['context'],
            example['answer'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )['input_ids'].view(-1).nonzero().squeeze().tolist()[-1]

        qa_inputs.append({
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'start_positions': start_positions,
            'end_positions': end_positions
        })

    return qa_inputs

qa_inputs = prepare_qa_data(qa_dataset)

# DataLoader for batch processing
train_dataloader = DataLoader(qa_inputs, batch_size=8, shuffle=True)

# optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# fine-tuning the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'start_positions': batch['start_positions'],
            'end_positions': batch['end_positions']
        }

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Avg Train Loss: {avg_train_loss}")

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')

question = input("Type your question: ")

# Tokenizing the input 
inputs = tokenizer(question, context, return_tensors='pt')

# Get model outputs
outputs = model(**inputs)

# Get answer start and end indices
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits)

# Convert indices to tokens and then to string
answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]
answer = tokenizer.decode(answer_tokens)

print("Answer:", answer)
