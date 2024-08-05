from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import AutoProcessor, AutoModelForCausalLM
from lavis.models import load_model_and_preprocess

from PIL import Image

""" Model specific functions
"""

class GlobalModel:
    def __init__(self, device="cpu") -> None:
        self.device=device

    def inference(self, inputs):
        raise NotImplementedError
    

class BLIPModel(GlobalModel):
    def __init__(self, device="cpu") -> None:
        super().__init__(device)
        self.model_blip_vqa2, self.vis_processors_blip_vqa2, self.txt_processors_blip_vqa2 = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

    def inference(self, inputs):
        raw_image, question, answer_candidates = inputs
        raw_image = raw_image.convert("RGB")
        image_blip_vqa2 = self.vis_processors_blip_vqa2["eval"](raw_image).unsqueeze(0).to(self.device)
        question_blip_vqa2 = self.txt_processors_blip_vqa2["eval"](question)
        samples_blip_vqa2 = {"image": image_blip_vqa2, "text_input": question_blip_vqa2}
        return self.model_blip_vqa2.predict_answers(samples_blip_vqa2, answer_list=answer_candidates, inference_method="rank")[0]

class VILTModel(GlobalModel):
    def __init__(self, device="cpu") -> None:
        super().__init__(device)
        self.processor_vilt = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        self.model_vilt = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").to(device)
        self.device=device

    def inference(self, inputs):
        raw_image, question, answer_candidates = inputs
        raw_image = raw_image.convert("RGB")

        encoding = self.processor_vilt(raw_image, question, return_tensors="pt")
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        outputs = self.model_vilt(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        answer = self.model_vilt.config.id2label[idx]
        if answer_candidates[0].lower() == answer:
            return answer_candidates[0]
        elif answer_candidates[1].lower() == answer:
            return answer_candidates[1]
        return self.model_vilt.config.id2label[idx]
        


# load vqa variants used in this experiment
# model_blip_vqa2, vis_processors_blip_vqa2, txt_processors_blip_vqa2 = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
# processor_vilt = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# model_vilt = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
# processor_git_vqav2 = AutoProcessor.from_pretrained("microsoft/git-base-vqav2")
# model_git_vqav2 = AutoModelForCausalLM.from_pretrained("microsoft/git-base-vqav2")


# # blip vqa
# def callblip(raw_image, question, answer_candidates):
#     raw_image = raw_image.convert("RGB")
#     image_blip_vqa2 = vis_processors_blip_vqa2["eval"](raw_image).unsqueeze(0).to(device)
#     question_blip_vqa2 = txt_processors_blip_vqa2["eval"](question)
#     samples_blip_vqa2 = {"image": image_blip_vqa2, "text_input": question_blip_vqa2}
#     return model_blip_vqa2.predict_answers(samples_blip_vqa2, answer_list=answer_candidates, inference_method="rank")[0]

# # vilt vqa
# def callvilt(pathofimag, question, answer_candidates):
#     raw_image = Image.open(pathofimag).convert("RGB")
#     encoding = processor_vilt(raw_image, question, return_tensors="pt")
#     outputs = model_vilt(**encoding)
#     logits = outputs.logits
#     idx = logits.argmax(-1).item()
#     return model_vilt.config.id2label[idx]

# # gitvqa
# def callgit(pathofimag, question, answer_candidates):
#     raw_image = Image.open(pathofimag).convert("RGB")
#     pixel_values = processor_git_vqav2(images=raw_image, return_tensors="pt").pixel_values
#     input_ids = processor_git_vqav2(text=question, add_special_tokens=False).input_ids
#     input_ids = [processor_git_vqav2.tokenizer.cls_token_id] + input_ids
#     input_ids = torch.tensor(input_ids).unsqueeze(0)
#     generated_ids = model_git_vqav2.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
#     return processor_git_vqav2.batch_decode(generated_ids, skip_special_tokens=True)[0].lstrip(question)

