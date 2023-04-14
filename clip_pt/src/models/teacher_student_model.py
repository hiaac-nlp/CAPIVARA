import torch.nn as nn
from transformers import AutoModel, CLIPTextModel


class TeacherStudentCLIPTBR(nn.Module):
    def __init__(
            self,
            teacher_version: str = "openai/clip-vit-base-patch32",
            student_version: str = "xlm-roberta-large",
    ):
        super().__init__()
        self.teacher = CLIPTextModel.from_pretrained(teacher_version,
                                                     cache_dir='/hahomes/gabriel.santos')
        self.student = AutoModel.from_pretrained(student_version,
                                                 cache_dir='/hahomes/gabriel.santos')

        self.teacher.gradient_checkpointing_enable()
        self.student.gradient_checkpointing_enable()

        self.transform = nn.Linear(
            self.student.pooler.dense.in_features,
            512,
            bias=False
        )

    def forward(self, data):
        source_lang_input, target_lang_input = data

        target_features = self.teacher(**target_lang_input)
        source_features = self.student(**source_lang_input)

        teacher_output = target_features.last_hidden_state
        eos = source_features.pooler_output  # pooled (EOS token) states
        student_output = self.transform(eos)

        return student_output, teacher_output
