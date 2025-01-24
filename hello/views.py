from django.http import HttpResponse
# from hello.models import YourModel


def home(request):

    # Fetch all records
    # records = YourModel.objects.all()
    # Fetch a single record by id
    # record = YourModel.objects.get(id=1)
    # Filter records
    # filtered_records = YourModel.objects.filter(field_name='value')

    return HttpResponse("Hello, Python / Django!")