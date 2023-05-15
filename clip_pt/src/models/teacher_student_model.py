import torch
import torch.nn as nn
from transformers import AutoModel, CLIPTextModel, CLIPTextModelWithProjection


class Student(nn.Module):
    def __init__(
            self,
            student_version: str = "neuralmind/bert-base-portuguese-cased",
    ):
        super().__init__()
        self.student = AutoModel.from_pretrained(student_version,
                                                 cache_dir='/hahomes/gabriel.santos')
        self.student.gradient_checkpointing_enable()
        self.pre_LN = nn.LayerNorm(self.student.pooler.dense.in_features, eps=1e-8)
        self.pooler = lambda x: x[:, 0]

        self.transform = nn.Linear(
            self.student.pooler.dense.in_features,
            512,
            bias=False
        )

    def forward(self, batch):
        output = self.student(**batch)
        sequence_output = self.pre_LN(output.last_hidden_state)
        pooled_output = self.pooler(sequence_output)
        return self.transform(pooled_output)


class TeacherStudentCLIPTBR(nn.Module):
    def __init__(
            self,
            teacher_version: str = "openai/clip-vit-base-patch32",
            student_version: str = "neuralmind/bert-base-portuguese-cased",
    ):
        super().__init__()
        self.teacher = CLIPTextModel.from_pretrained(teacher_version,
                                                     cache_dir='/hahomes/gabriel.santos')
        # freeze teacher params
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = Student(student_version)


    def forward(self, data):
        teacher_input, student_input = data

        target_features = self.teacher(**teacher_input)
        teacher_output = target_features.pooler_output

        student_output = self.student(student_input)

        return teacher_output, student_output


class TeacherStudent_mCLIP(nn.Module):
    def __init__(
            self,
            teacher_version: str = "openai/clip-vit-base-patch32",
            student_version: str = "neuralmind/bert-base-portuguese-cased",
    ):
        super().__init__()
        self.teacher = CLIPTextModelWithProjection.from_pretrained(teacher_version,
                                                     cache_dir='/hahomes/gabriel.santos')
        # freeze teacher params
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = Student(student_version)

    def forward(self, data):
        teacher_input, student_input = data

        target_features = self.teacher(**teacher_input)
        teacher_output = target_features.text_embeds

        student_output = self.student(student_input)

        return teacher_output, student_output

class Student_MeanPooling(nn.Module):
    def __init__(
            self,
            student_version: str = "neuralmind/bert-base-portuguese-cased",
    ):
        super().__init__()
        self.student = AutoModel.from_pretrained(student_version,
                                                 cache_dir='/hahomes/gabriel.santos')
        self.student.gradient_checkpointing_enable()
        self.transform = nn.Linear(
            self.student.pooler.dense.in_features,
            512,
            bias=False
        )

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, batch):
        output = self.student(**batch)
        return self.transform(self.mean_pooling(output, batch["attention_mask"]))


class TeacherStudent_MeanPooling(nn.Module):
    def __init__(
            self,
            teacher_version: str = "openai/clip-vit-base-patch32",
            student_version: str = "neuralmind/bert-base-portuguese-cased",
    ):
        super().__init__()
        self.teacher = CLIPTextModelWithProjection.from_pretrained(teacher_version,
                                                     cache_dir='/hahomes/gabriel.santos')
        # freeze teacher params
        for param in self.teacher.parameters():
            param.requires_grad = False

        self.student = Student_MeanPooling(student_version)

    def forward(self, data):
        teacher_input, student_input = data

        target_features = self.teacher(**teacher_input)
        teacher_output = target_features.text_embeds

        student_output = self.student(student_input)

        return teacher_output, student_output
